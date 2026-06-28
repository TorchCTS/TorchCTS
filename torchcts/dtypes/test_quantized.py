# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import torch
from torchcts.core.quantized_decoders import load_custom_container_decoder
from torchcts.core.device import synchronize


# ═══════════════════════════════════════════════════════════════════
# Scale modes — matches metalcore ScaleMode enum
# ═══════════════════════════════════════════════════════════════════
class ScaleMode:
    NONE = 0
    PER_TENSOR = 1
    PER_CHANNEL = 2
    BLOCK = 3


# ═══════════════════════════════════════════════════════════════════
# Codec registry
# ═══════════════════════════════════════════════════════════════════
class ContainerCodec:
    """Base class for container format pack/unpack codecs."""
    bits: int = 8
    scale_mode: int = ScaleMode.NONE
    has_zero_point: bool = False
    block_k: int = 0

    def generate(self, n):
        raise NotImplementedError

    def pack(self, vals):
        raise NotImplementedError

    def unpack(self, packed, n):
        raise NotImplementedError

    def generate_scale(self, n):
        """Generate scale tensor appropriate for this format.
        Returns (scale, zero_point) where zero_point may be None."""
        if self.scale_mode == ScaleMode.NONE:
            return None, None
        elif self.scale_mode == ScaleMode.PER_TENSOR:
            scale = torch.tensor([torch.randn(1).abs().item() + 0.1], dtype=torch.float32)
            zp = torch.zeros(1, dtype=torch.float32) if self.has_zero_point else None
            return scale, zp
        elif self.scale_mode == ScaleMode.PER_CHANNEL:
            n_channels = max(1, n // 128)
            scale = torch.randn(n_channels, dtype=torch.float32).abs() + 0.1
            zp = torch.zeros(n_channels, dtype=torch.float32) if self.has_zero_point else None
            return scale, zp
        elif self.scale_mode == ScaleMode.BLOCK:
            n_blocks = max(1, n // self.block_k)
            scale = torch.randn(n_blocks, dtype=torch.float32).abs() + 0.1
            zp = torch.zeros(n_blocks, dtype=torch.float32) if self.has_zero_point else None
            return scale, zp
        return None, None


_CODEC_REGISTRY = {}


def register_codec(name, **kwargs):
    """Decorator to register a ContainerCodec subclass.

    Sets attributes on the INSTANCE, not the class, so that loops
    registering the same class with different configs don't clobber
    each other.
    """
    def decorator(cls):
        instance = cls()
        for k, v in kwargs.items():
            setattr(instance, k, v)
        _CODEC_REGISTRY[name] = instance
        return cls
    return decorator


# ═══════════════════════════════════════════════════════════════════
# 2-bit codecs
# ═══════════════════════════════════════════════════════════════════
@register_codec("int2_ternary", bits=2, scale_mode=ScaleMode.PER_CHANNEL)
class Int2TernaryCodec(ContainerCodec):
    def generate(self, n):
        return torch.randint(-1, 2, (n,), dtype=torch.int8).float()

    def pack(self, vals):
        shifted = (vals.to(torch.int8) + 1).to(torch.uint8)
        n = len(shifted)
        assert n % 4 == 0
        packed = torch.zeros(n // 4, dtype=torch.uint8)
        for i in range(4):
            packed |= (shifted[i::4] << (i * 2))
        return packed

    def unpack(self, packed, n):
        out = torch.zeros(n, dtype=torch.int8)
        for i in range(4):
            out[i::4] = ((packed >> (i * 2)) & 0x3).to(torch.int8) - 1
        return out.float()


# ═══════════════════════════════════════════════════════════════════
# 4-bit codecs
# ═══════════════════════════════════════════════════════════════════
@register_codec("int4_symmetric", bits=4, scale_mode=ScaleMode.PER_CHANNEL)
class Int4SymCodec(ContainerCodec):
    def generate(self, n):
        return torch.randint(-8, 8, (n,), dtype=torch.int8).float()

    def pack(self, vals):
        shifted = (vals.to(torch.int8) + 8).to(torch.uint8)
        return (shifted[0::2] << 4) | shifted[1::2]

    def unpack(self, packed, n):
        high = (packed >> 4).to(torch.int8) - 8
        low = (packed & 0x0F).to(torch.int8) - 8
        out = torch.zeros(n, dtype=torch.int8)
        out[0::2] = high
        out[1::2] = low
        return out.float()


@register_codec("int4_asymmetric", bits=4, scale_mode=ScaleMode.PER_CHANNEL, has_zero_point=True)
class Int4AsymCodec(ContainerCodec):
    def generate(self, n):
        return torch.randint(0, 16, (n,), dtype=torch.uint8).float()

    def pack(self, vals):
        v = vals.to(torch.uint8)
        return (v[0::2] << 4) | v[1::2]

    def unpack(self, packed, n):
        out = torch.zeros(n, dtype=torch.uint8)
        out[0::2] = packed >> 4
        out[1::2] = packed & 0x0F
        return out.float()


@register_codec("uint4", bits=4, scale_mode=ScaleMode.PER_CHANNEL)
class Uint4Codec(ContainerCodec):
    def generate(self, n):
        return torch.randint(0, 16, (n,), dtype=torch.uint8).float()

    def pack(self, vals):
        v = vals.to(torch.uint8)
        return (v[0::2] << 4) | v[1::2]

    def unpack(self, packed, n):
        out = torch.zeros(n, dtype=torch.uint8)
        out[0::2] = packed >> 4
        out[1::2] = packed & 0x0F
        return out.float()


_NF4_CODEBOOK = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]


@register_codec("nf4", bits=4, scale_mode=ScaleMode.BLOCK, block_k=64)
class Nf4Codec(ContainerCodec):
    def generate(self, n):
        indices = torch.randint(0, 16, (n,), dtype=torch.uint8)
        cb = torch.tensor(_NF4_CODEBOOK, dtype=torch.float32)
        return cb[indices.long()]

    def pack(self, vals):
        cb = torch.tensor(_NF4_CODEBOOK, dtype=torch.float32)
        indices = torch.argmin(torch.abs(vals.unsqueeze(1) - cb.unsqueeze(0)), dim=1).to(torch.uint8)
        return (indices[0::2] << 4) | indices[1::2]

    def unpack(self, packed, n):
        cb = torch.tensor(_NF4_CODEBOOK, dtype=torch.float32)
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        out_idx = torch.zeros(n, dtype=torch.long)
        out_idx[0::2] = high
        out_idx[1::2] = low
        return cb[out_idx]


_FP4_BNB_CODEBOOK = [
    0.0, 0.0625, 0.25, 0.3125, 0.5, 0.625, 0.75, 0.875,
    -0.0, -0.0625, -0.25, -0.3125, -0.5, -0.625, -0.75, -0.875,
]


@register_codec("fp4_bnb", bits=4, scale_mode=ScaleMode.BLOCK, block_k=64)
class Fp4BnbCodec(ContainerCodec):
    def generate(self, n):
        indices = torch.randint(0, 16, (n,), dtype=torch.uint8)
        cb = torch.tensor(_FP4_BNB_CODEBOOK, dtype=torch.float32)
        return cb[indices.long()]

    def pack(self, vals):
        cb = torch.tensor(_FP4_BNB_CODEBOOK, dtype=torch.float32)
        indices = torch.argmin(torch.abs(vals.unsqueeze(1) - cb.unsqueeze(0)), dim=1).to(torch.uint8)
        return (indices[0::2] << 4) | indices[1::2]

    def unpack(self, packed, n):
        cb = torch.tensor(_FP4_BNB_CODEBOOK, dtype=torch.float32)
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        out_idx = torch.zeros(n, dtype=torch.long)
        out_idx[0::2] = high
        out_idx[1::2] = low
        return cb[out_idx]


@register_codec("fp4_e2m1", bits=4, scale_mode=ScaleMode.BLOCK, block_k=32)
class Fp4E2M1Codec(ContainerCodec):
    def generate(self, n):
        return torch.randint(0, 16, (n,), dtype=torch.uint8).float()

    def pack(self, vals):
        v = vals.to(torch.uint8)
        return (v[0::2] << 4) | v[1::2]

    def unpack(self, packed, n):
        out = torch.zeros(n, dtype=torch.uint8)
        out[0::2] = packed >> 4
        out[1::2] = packed & 0x0F
        return out.float()


class _Generic4BitCodec(ContainerCodec):
    """Generic 4-bit index packing for formats with backend-specific decode."""
    def generate(self, n):
        return torch.randint(0, 16, (n,), dtype=torch.uint8).float()
    def pack(self, vals):
        v = vals.to(torch.uint8)
        return (v[0::2] << 4) | v[1::2]
    def unpack(self, packed, n):
        out = torch.zeros(n, dtype=torch.uint8)
        out[0::2] = packed >> 4
        out[1::2] = packed & 0x0F
        return out.float()

for _name, _sm, _bk in [
    ("af4", ScaleMode.BLOCK, 64),
    ("mxfp4", ScaleMode.BLOCK, 32),
    ("nvfp4", ScaleMode.BLOCK, 128),
]:
    register_codec(_name, bits=4, scale_mode=_sm, block_k=_bk)(_Generic4BitCodec)


# ═══════════════════════════════════════════════════════════════════
# 6-bit codecs — 4 elements packed into 3 bytes
# ═══════════════════════════════════════════════════════════════════
class _Base6BitCodec(ContainerCodec):
    def generate(self, n):
        return torch.randint(0, 64, (n,), dtype=torch.uint8).float()

    def pack(self, vals):
        v = vals.to(torch.uint8)
        n = len(v)
        assert n % 4 == 0
        groups = n // 4
        packed = torch.zeros(groups * 3, dtype=torch.uint8)
        for g in range(groups):
            a, b, c, d = v[g*4], v[g*4+1], v[g*4+2], v[g*4+3]
            packed[g*3 + 0] = (a << 2) | (b >> 4)
            packed[g*3 + 1] = ((b & 0xF) << 4) | (c >> 2)
            packed[g*3 + 2] = ((c & 0x3) << 6) | d
        return packed

    def unpack(self, packed, n):
        groups = n // 4
        out = torch.zeros(n, dtype=torch.uint8)
        for g in range(groups):
            b0, b1, b2 = packed[g*3].item(), packed[g*3+1].item(), packed[g*3+2].item()
            out[g*4 + 0] = (b0 >> 2) & 0x3F
            out[g*4 + 1] = ((b0 & 0x3) << 4) | (b1 >> 4)
            out[g*4 + 2] = ((b1 & 0xF) << 2) | (b2 >> 6)
            out[g*4 + 3] = b2 & 0x3F
        return out.float()

for _name, _sm, _bk in [
    ("fp6_e3m2", ScaleMode.PER_CHANNEL, 0),
    ("fp6_e2m3", ScaleMode.PER_CHANNEL, 0),
    ("mxfp6_e3m2", ScaleMode.BLOCK, 32),
    ("mxfp6_e2m3", ScaleMode.BLOCK, 32),
]:
    register_codec(_name, bits=6, scale_mode=_sm, block_k=_bk)(_Base6BitCodec)


# ═══════════════════════════════════════════════════════════════════
# 8-bit codecs — 1:1 packing
# ═══════════════════════════════════════════════════════════════════
@register_codec("int8_symmetric", bits=8, scale_mode=ScaleMode.PER_CHANNEL)
class Int8SymCodec(ContainerCodec):
    def generate(self, n):
        return torch.randint(-128, 128, (n,), dtype=torch.int8).float()
    def pack(self, vals):
        return vals.to(torch.int8).view(torch.uint8)
    def unpack(self, packed, n):
        return packed.view(torch.int8).float()


@register_codec("int8_asymmetric", bits=8, scale_mode=ScaleMode.PER_CHANNEL, has_zero_point=True)
class Int8AsymCodec(ContainerCodec):
    def generate(self, n):
        return torch.randint(0, 256, (n,), dtype=torch.uint8).float()
    def pack(self, vals):
        return vals.to(torch.uint8)
    def unpack(self, packed, n):
        return packed.float()


@register_codec("uint8", bits=8, scale_mode=ScaleMode.PER_TENSOR)
class Uint8Codec(ContainerCodec):
    def generate(self, n):
        return torch.randint(0, 256, (n,), dtype=torch.uint8).float()
    def pack(self, vals):
        return vals.to(torch.uint8)
    def unpack(self, packed, n):
        return packed.float()


class _Identity8BitCodec(ContainerCodec):
    """1:1 packing for 8-bit FP and MX formats."""
    def generate(self, n):
        return torch.randint(0, 256, (n,), dtype=torch.uint8).float()
    def pack(self, vals):
        return vals.to(torch.uint8)
    def unpack(self, packed, n):
        return packed.float()

for _name, _sm, _bk in [
    ("fp8_e4m3fn", ScaleMode.PER_TENSOR, 0),
    ("fp8_e5m2", ScaleMode.PER_TENSOR, 0),
    ("fp8_e4m3fnuz", ScaleMode.PER_TENSOR, 0),
    ("fp8_e5m2fnuz", ScaleMode.PER_TENSOR, 0),
    ("mxfp8_e4m3", ScaleMode.BLOCK, 32),
    ("mxfp8_e5m2", ScaleMode.BLOCK, 32),
    ("mxint8", ScaleMode.BLOCK, 32),
]:
    register_codec(_name, bits=8, scale_mode=_sm, block_k=_bk)(_Identity8BitCodec)


@register_codec("e8m0fnu", bits=8, scale_mode=ScaleMode.NONE)
class E8m0fnuCodec(ContainerCodec):
    """E8M0 is a scale format itself — no separate scale."""
    def generate(self, n):
        return torch.randint(0, 256, (n,), dtype=torch.uint8).float()
    def pack(self, vals):
        return vals.to(torch.uint8)
    def unpack(self, packed, n):
        return packed.float()


# ═══════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.medium
@pytest.mark.requires("quantized_container_plumbing")
@pytest.mark.covers("aten::_to_copy")
@pytest.mark.covers_category("quantized_container_plumbing")
@pytest.mark.parametrize("packing", list(_CODEC_REGISTRY.keys()))
def test_quantized_plumbing(packing, device, manifest):
    """Verify pack/unpack round-trips and scale tensor device transfers."""
    supported = manifest.get("supported_container_formats", {})
    if not supported.get(packing, False):
        pytest.skip(f"Container format '{packing}' not in supported_container_formats")

    codec = _CODEC_REGISTRY[packing]
    n_elements = 128

    # 1. Generate reference values and pack
    original = codec.generate(n_elements)
    packed_cpu = codec.pack(original)

    # 2. Generate scale and zero_point
    scale, zero_point = codec.generate_scale(n_elements)

    # 3. Round-trip packed data through device
    packed_dev = packed_cpu.to(device)
    synchronize(device)
    packed_roundtrip = packed_dev.cpu()
    assert torch.equal(packed_roundtrip, packed_cpu), (
        f"Packed data corrupted during device round-trip for '{packing}'"
    )

    # 4. Round-trip scale through device (if applicable)
    if scale is not None:
        scale_dev = scale.to(device)
        synchronize(device)
        scale_roundtrip = scale_dev.cpu()
        assert torch.equal(scale_roundtrip, scale), (
            f"Scale tensor corrupted during device round-trip for '{packing}'"
        )

    # 5. Round-trip zero_point through device (if applicable)
    if zero_point is not None:
        zp_dev = zero_point.to(device)
        synchronize(device)
        zp_roundtrip = zp_dev.cpu()
        assert torch.equal(zp_roundtrip, zero_point), (
            f"Zero-point tensor corrupted during device round-trip for '{packing}'"
        )

    # 6. Verify reference codec round-trip
    unpacked = codec.unpack(packed_cpu, n_elements)
    assert torch.equal(unpacked, original), (
        f"Reference pack/unpack round-trip failed for '{packing}'"
    )


def _run_custom_decoder_case(packing, decoder, device, compare):
    codec = _CODEC_REGISTRY[packing]
    n_elements = 128

    original = codec.generate(n_elements)
    packed_cpu = codec.pack(original)
    scale, zero_point = codec.generate_scale(n_elements)

    packed_dev = packed_cpu.to(device)
    scale_dev = scale.to(device) if scale is not None else None
    zero_point_dev = zero_point.to(device) if zero_point is not None else None
    decoded = decoder(
        packed_dev,
        scale_dev,
        zero_point_dev,
        (n_elements,),
        torch.float32,
        device,
    )
    synchronize(device)

    if not isinstance(decoded, torch.Tensor):
        raise AssertionError(f"Custom decoder for '{packing}' returned {type(decoded).__name__}, not torch.Tensor")
    expected = codec.unpack(packed_cpu, n_elements).to(torch.float32)
    compare(decoded, expected, category="quant_decode", dtype=torch.float32)


@pytest.mark.medium
@pytest.mark.requires("custom_quantized_decode")
@pytest.mark.covers("aten::_to_copy")
@pytest.mark.covers_category("custom_quantized_decode")
@pytest.mark.parametrize("packing", list(_CODEC_REGISTRY.keys()))
def test_custom_quantized_decoder(packing, device, manifest, compare):
    """Verify user-provided semantic decoders against TorchCTS reference codecs."""
    supported = manifest.get("supported_container_formats", {})
    if not supported.get(packing, False):
        pytest.skip(f"Container format '{packing}' not in supported_container_formats")

    decoder_specs = manifest.get("custom_container_decoders", {})
    decoder_spec = decoder_specs.get(packing)
    if not decoder_spec:
        pytest.skip(f"No custom decoder declared for container format '{packing}'")

    decoder = load_custom_container_decoder(decoder_spec)
    _run_custom_decoder_case(packing, decoder, device, compare)

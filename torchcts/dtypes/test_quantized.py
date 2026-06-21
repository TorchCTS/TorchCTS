import pytest
import torch
from torchcts.core.device import synchronize

@pytest.mark.medium
@pytest.mark.requires("quantized")
@pytest.mark.parametrize("packing", ["int4"])
def test_quantized_plumbing(packing, device, manifest):
    # Quantized formats are packed into uint8 carrier tensors.
    # We test that custom packing/unpacking utilities round-trip correctly.
    # Define sym/asym int4 packing
    def pack_int4(x):
        # x is int8 tensor, packed two 4-bit values per uint8
        x_clamped = torch.clamp(x, -8, 7)
        # Shift values to 0-15 range
        x_shifted = x_clamped + 8
        high = x_shifted[::2] << 4
        low = x_shifted[1::2]
        packed = (high | low).to(torch.uint8)
        return packed
        
    def unpack_int4(packed):
        # packed is uint8 tensor
        high = (packed >> 4).to(torch.int8) - 8
        low = (packed & 0x0F).to(torch.int8) - 8
        unpacked = torch.zeros(len(packed) * 2, dtype=torch.int8)
        unpacked[::2] = high
        unpacked[1::2] = low
        return unpacked
        
    x_cpu = torch.randint(-8, 8, (128,), dtype=torch.int8)
    # Only "int4" is parametrized; no need for a runtime format check.
    packed_cpu = pack_int4(x_cpu)
    
    packed_dev = packed_cpu.to(device)
    synchronize(device)
    
    # Check that they match
    assert torch.equal(packed_dev.cpu(), packed_cpu)
    
    # Unpack and verify
    if packing == "int4":
        unpacked_dev = unpack_int4(packed_dev.cpu())
    assert torch.equal(unpacked_dev, x_cpu)

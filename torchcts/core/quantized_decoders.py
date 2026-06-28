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

import importlib
import re


KNOWN_CONTAINER_FORMATS = frozenset({
    "int2_ternary",
    "int4_symmetric",
    "int4_asymmetric",
    "uint4",
    "nf4",
    "fp4_e2m1",
    "fp4_bnb",
    "af4",
    "mxfp4",
    "nvfp4",
    "fp6_e3m2",
    "fp6_e2m3",
    "mxfp6_e3m2",
    "mxfp6_e2m3",
    "fp8_e4m3fn",
    "fp8_e5m2",
    "fp8_e4m3fnuz",
    "fp8_e5m2fnuz",
    "int8_symmetric",
    "int8_asymmetric",
    "uint8",
    "e8m0fnu",
    "mxfp8_e4m3",
    "mxfp8_e5m2",
    "mxint8",
})

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_decoder_spec(spec):
    if not isinstance(spec, str) or ":" not in spec:
        return "decoder path must be a string in 'module:function' form"

    module_name, attr_path = spec.split(":", 1)
    module_parts = module_name.split(".")
    attr_parts = attr_path.split(".")
    if (
        not module_name
        or not attr_path
        or any(not _IDENTIFIER_RE.match(part) for part in module_parts)
        or any(not _IDENTIFIER_RE.match(part) for part in attr_parts)
    ):
        return "decoder path must be a valid 'module:function' import path"

    return None


def load_custom_container_decoder(spec):
    error = validate_decoder_spec(spec)
    if error:
        raise ValueError(error)

    module_name, attr_path = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj = module
    for part in attr_path.split("."):
        obj = getattr(obj, part)

    if not callable(obj):
        raise TypeError(f"custom container decoder {spec!r} is not callable")
    return obj

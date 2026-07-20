# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

import pytest
import torch
import torch.nn.functional as F

from torchcts.core.device import synchronize


@pytest.mark.covers("aten::grid_sampler_2d")
@pytest.mark.covers("aten::grid_sampler_3d")
@pytest.mark.parametrize(
    "input_shape, grid_shape",
    (
        ((1, 1, 3, 4), (1, 2, 2, 2)),
        ((1, 1, 2, 3, 4), (1, 1, 2, 2, 3)),
    ),
    ids=("2d", "3d"),
)
def test_r086_grid_sample_nonfinite_coordinate_contract(
    input_shape,
    grid_shape,
    device,
    compare,
):
    input_cpu = torch.linspace(-1, 1, int(torch.tensor(input_shape).prod())).reshape(input_shape)
    grid_cpu = torch.zeros(grid_shape)

    nan_grid = grid_cpu.clone()
    nan_grid.reshape(-1, grid_shape[-1])[0, 0] = float("nan")
    expected_nan = F.grid_sample(
        input_cpu,
        torch.nan_to_num(nan_grid, nan=-1.0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    inf_grid = grid_cpu.clone()
    inf_grid.reshape(-1, grid_shape[-1])[0, 0] = float("inf")
    safe_grid = torch.where(torch.isfinite(inf_grid), inf_grid, torch.full_like(inf_grid, -100.0))
    expected_inf = F.grid_sample(
        input_cpu,
        safe_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    # CPU is the upstream implementation under notification.  CPU validation
    # proves the finite surrogate references are executable without treating
    # the affected nonfinite CPU kernel as its own oracle.
    if torch.device(device).type == "cpu":
        assert torch.isfinite(expected_nan).all()
        assert torch.isfinite(expected_inf).all()
        return

    input_device = input_cpu.to(device)
    actual_nan = F.grid_sample(
        input_device,
        nan_grid.to(device),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    actual_inf = F.grid_sample(
        input_device,
        inf_grid.to(device),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    synchronize(device)

    compare(actual_nan, expected_nan, category="grid_sample", dtype=torch.float32)
    compare(actual_inf, expected_inf, category="grid_sample", dtype=torch.float32)

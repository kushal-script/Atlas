"""Parameter definitions for layout geometry and SEM image formation.

Every numeric range here is grounded in public literature, see docs/citations.md
for the reference list keyed by parameter group.
"""

from dataclasses import dataclass, field


MATERIAL_STI = 0
MATERIAL_SILICON = 1
MATERIAL_GATE = 2
MATERIAL_TUNGSTEN = 3
MATERIAL_NITRIDE = 4

BASE_SE_YIELD = {
    MATERIAL_STI: 1.45,
    MATERIAL_SILICON: 1.00,
    MATERIAL_GATE: 1.20,
    MATERIAL_TUNGSTEN: 1.50,
    MATERIAL_NITRIDE: 1.30,
}


@dataclass
class CanvasParams:
    extent_nm: float = 11200.0
    pixel_nm: float = 2.0
    sidewall_sigma_nm: float = 2.5
    max_tilt_deg: float = 82.0
    detector_asymmetry: tuple = (0.05, 0.18)
    yield_jitter: float = 0.10

    @property
    def size_px(self) -> int:
        return int(round(self.extent_nm / self.pixel_nm))


@dataclass
class DramParams:
    feature_nm: tuple = (16.0, 22.0)
    wl_duty: tuple = (0.42, 0.55)
    bl_duty: tuple = (0.38, 0.50)
    contact_radius_f: tuple = (0.70, 0.90)
    contact_cd_sigma: float = 0.04
    contact_missing_prob: tuple = (0.002, 0.012)
    mat_width_nm: tuple = (5500.0, 9000.0)
    mat_height_nm: tuple = (4500.0, 8000.0)
    sa_stripe_nm: tuple = (380.0, 560.0)
    swd_stripe_nm: tuple = (300.0, 460.0)
    ler_sigma_nm: tuple = (1.2, 2.4)
    ler_corr_nm: tuple = (18.0, 40.0)
    wl_height_nm: float = 30.0
    bl_height_nm: float = 55.0
    contact_height_nm: float = 78.0


@dataclass
class CaptureParams:
    pixel_nm: float
    supersample: int
    psf_sigma_nm: tuple
    astig_ratio: tuple
    dose_e: tuple
    read_noise_e: tuple
    drift_total_px: tuple
    jitter_sigma_px: tuple
    vib_amp_px: tuple
    charging_amp: tuple
    charging_scale_nm: tuple = (120.0, 320.0)
    out_px: int = 1000


REFERENCE_CAPTURE = CaptureParams(
    pixel_nm=1.0,
    supersample=1,
    psf_sigma_nm=(1.5, 2.6),
    astig_ratio=(1.0, 1.25),
    dose_e=(700.0, 2000.0),
    read_noise_e=(2.0, 6.0),
    drift_total_px=(0.0, 3.0),
    jitter_sigma_px=(0.05, 0.35),
    vib_amp_px=(0.0, 0.30),
    charging_amp=(0.02, 0.06),
)

SEARCH_CAPTURE = CaptureParams(
    pixel_nm=10.0,
    supersample=5,
    psf_sigma_nm=(4.0, 12.0),
    astig_ratio=(1.0, 1.35),
    dose_e=(80.0, 300.0),
    read_noise_e=(2.0, 8.0),
    drift_total_px=(0.0, 1.2),
    jitter_sigma_px=(0.03, 0.25),
    vib_amp_px=(0.0, 0.25),
    charging_amp=(0.03, 0.09),
)


@dataclass
class PoseParams:
    rotation_deg_sigma: float = 0.8
    rotation_deg_max: float = 2.0
    scale_err_sigma: float = 0.008
    scale_err_max: float = 0.018
    search_center_jitter_nm: float = 200.0
    ref_margin_px: float = 70.0
    # Phase 2 draws the pose from disclosed ranges rather than as a small error
    # around a nominal ten to one: the zoom ratio is uniform over the range and
    # the rotation of the reference as it appears in the search image is uniform
    # over plus or minus the maximum. These are only read when phase2 is set, so
    # the Phase 1 behaviour and every recorded Phase 1 result are unaffected.
    zoom_min: float = 8.0
    zoom_max: float = 12.0
    rel_rotation_deg_max: float = 5.0


@dataclass
class GeneratorConfig:
    canvas: CanvasParams = field(default_factory=CanvasParams)
    dram: DramParams = field(default_factory=DramParams)
    reference: CaptureParams = field(default_factory=lambda: REFERENCE_CAPTURE)
    search: CaptureParams = field(default_factory=lambda: SEARCH_CAPTURE)
    pose: PoseParams = field(default_factory=PoseParams)
    placement_mix: tuple = (("uniform", 0.5), ("deep_array", 0.25), ("near_boundary", 0.25))
    # Phase 2: draw the pose from the disclosed ranges instead of jittering
    # around a fixed ten to one.
    phase2: bool = False

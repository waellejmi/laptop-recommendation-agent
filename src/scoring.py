import pandas as pd


# These profiles should be trained in a real world scenario but due to no labeled data i decided to hardcode them.(they are a bit dumb)
def get_weights(usage_profile: str, user_emphasis: list | None = None):
    base = {
        "gaming": {
            "cpu_tier": 0.30,
            "gpu_tier": 0.45,
            "ram": 0.10,
            "ssd_present": 0.05,
            "price": 0.10,
        },
        "student": {
            "cpu_tier": 0.40,
            "gpu_tier": 0.15,
            "ram": 0.20,
            "ssd_present": 0.05,
            "price": 0.20,
        },
        "basic": {
            "cpu_tier": 0.35,
            "gpu_tier": 0.05,
            "ram": 0.25,
            "ssd_present": 0.05,
            "price": 0.20,
        },
        "workstation": {
            "cpu_tier": 0.55,
            "gpu_tier": 0.25,
            "ram": 0.20,
            "ssd_present": 0.01,
            "price": 0.04,
        },
    }[usage_profile]

    if user_emphasis:
        for feature in user_emphasis:
            if feature in base:
                base[feature] *= 1.5
        total = sum(base.values())
        base = {k: v / total for k, v in base.items()}

    return base


def compute_scores(df: pd.DataFrame, weights: dict):
    df = df.copy()

    df["final_score"] = (
        df["norm_cpu"] * weights["cpu_tier"]
        + df["norm_gpu"] * weights["gpu_tier"]
        + df["norm_ram"] * weights["ram"]
        + df["ssd_present"] * weights["ssd_present"]
        + df["norm_price"] * weights["price"]
    )

    return df.sort_values("final_score", ascending=False)

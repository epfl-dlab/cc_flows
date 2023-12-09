import hydra


def get_config(config_name, overrides=[], work_dir="../../", data_dir="../../data/", configs_folder="../../configs"):
    if work_dir is not None:
        overrides.append(f"work_dir={work_dir}")
    if data_dir is not None:
        overrides.append(f"data_dir={data_dir}")

    with hydra.initialize(version_base="1.2", config_path=configs_folder):
        config = hydra.compose(config_name=config_name, overrides=overrides)

    return config

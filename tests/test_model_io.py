import json

from library import model_io


def test_remove_caption_metadata_preserves_unrelated_metadata():
    datasets = [
        {
            "resolution": [1024, 1024],
            "tag_frequency": {"characters": {"example tag": 3}},
            "subsets": [
                {
                    "img_count": 3,
                    "shuffle_caption": True,
                    "keep_tokens": 1,
                    "keep_tokens_separator": "|||",
                    "secondary_separator": ";",
                    "enable_wildcard": True,
                    "caption_prefix": "prefix",
                    "caption_suffix": "suffix",
                    "class_tokens": "example class",
                    "image_dir": "characters",
                }
            ],
        }
    ]
    metadata = {
        "ss_network_dim": "16",
        "ss_caption_dropout_rate": "0.1",
        "ss_caption_dropout_every_n_epochs": "2",
        "ss_caption_tag_dropout_rate": "0.2",
        "ss_shuffle_caption": "True",
        "ss_keep_tokens": "1",
        "ss_tag_frequency": json.dumps({"characters": {"example tag": 3}}),
        "ss_datasets": json.dumps(datasets),
    }

    filtered = model_io.remove_caption_metadata(metadata)

    assert filtered["ss_network_dim"] == "16"
    assert not model_io.CAPTION_METADATA_KEYS.intersection(filtered)

    filtered_datasets = json.loads(filtered["ss_datasets"])
    assert filtered_datasets == [
        {
            "resolution": [1024, 1024],
            "subsets": [{"img_count": 3, "image_dir": "characters"}],
        }
    ]

    # Filtering a copy keeps the full in-memory metadata available to callers.
    assert metadata["ss_tag_frequency"]
    assert json.loads(metadata["ss_datasets"])[0]["tag_frequency"]


def test_remove_caption_metadata_without_dataset_metadata():
    metadata = {"ss_network_dim": "16", "ss_tag_frequency": "{}"}

    assert model_io.remove_caption_metadata(metadata) == {"ss_network_dim": "16"}

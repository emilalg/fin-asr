from datasets import load_dataset, get_dataset_split_names, load_dataset_builder

print(get_dataset_split_names("mozilla-foundation/common_voice_7_0"))
print(get_dataset_split_names("mozilla-foundation/common_voice_7_0", "fi"))

builder = load_dataset_builder("mozilla-foundation/common_voice_7_0", "fi", token=True)
print(builder.keys())
print(f"Dataset name: {builder.name}")
print(f"Dataset version: {builder.version}")
print(f"Dataset description: {builder.description}")
print(f"Dataset features: {builder.features}")
print(f"Dataset splits: {builder.info.splits}")

dataset_dict = load_dataset(
    "mozilla-foundation/common_voice_7_0",
    "fi", 
    split='test',
    cache_dir='data/hf',
    token=True
)

print(dataset_dict)
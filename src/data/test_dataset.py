from dataset import HAM10000Dataset, train_transform


dataset = HAM10000Dataset(
    "data/processed/train.csv",
    transform=train_transform
)

print("Dataset size:", len(dataset))

image, metadata, label = dataset[0]

print("\nImage:")
print("Shape:", image.shape)
print("Type:", image.dtype)

print("\nMetadata:")
print(metadata)
print("Shape:", metadata.shape)

print("\nLabel:")
print(label)
print("Class index:", label.item())
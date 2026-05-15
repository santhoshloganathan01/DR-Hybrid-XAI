from torch.utils.data import DataLoader
from training.dataset import FundusDataset

dataset = FundusDataset(
    csv_file="data/labels.csv",
    img_dir="data/images"
)

print("Total samples:", len(dataset))

img, label = dataset[0]

print("Single image shape:", img.shape)
print("Label:", label)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

batch_imgs, batch_labels = next(iter(loader))

print("Batch shape:", batch_imgs.shape)
print("Batch labels shape:", batch_labels.shape)

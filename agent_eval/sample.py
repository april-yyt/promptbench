import promptbench as pb

def load_dataset(dataset_name):
    return pb.DatasetLoader.load_dataset(dataset_name)

# Load datasets
train_data = load_dataset('gsm8k')
test_data = load_dataset('gsm8k')  # You might want to use a different split for test data

# Check if data is loaded
print(f"Train data length: {len(train_data)}")
print(f"Test data length: {len(test_data)}")

# Print a sample
print("Sample from train data:", train_data[0] if train_data else "No data")
print("Sample from test data:", test_data[0] if test_data else "No data")

# Rest of your evaluation code...
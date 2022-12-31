import argparse
print("Make America Great Again")
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='run_001.pth')
opt = parser.parse_args()
print(opt.model_name)

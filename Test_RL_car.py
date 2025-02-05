import subprocess

def main():
    script_path = "low_rank_rl/car/main_car.py"
    
    try:
        subprocess.run(["python3", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing {script_path}: {e}")
    except FileNotFoundError:
        print("Python3 interpreter not found. Please ensure Python3 is installed and in your PATH.")

if __name__ == "__main__":
    main()
from .cli import Cli
import fire

if __name__ == "__main__":
    try:
        fire.Fire(Cli)
    except Exception as error:
        print(f"Error: {error}")
    except KeyboardInterrupt:
        print("the programme was interrupted by user")

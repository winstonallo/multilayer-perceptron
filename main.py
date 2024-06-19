from src.user_interface import CommandLineInterface


def main():
    try:
        cmd_line_interface = CommandLineInterface()
        cmd_line_interface.run()
    except Exception as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()

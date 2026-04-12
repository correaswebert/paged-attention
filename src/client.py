import sys

import requests


def run_cli():
    print("=" * 50)
    print("Paged Attention CLI Client")
    print("Connecting to http://localhost:8000/chat")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 50)

    url = "http://localhost:8000/chat"

    while True:
        try:
            user_input = input("\nUser> ")

            if user_input.strip().lower() in ["quit", "exit"]:
                print("Exiting...")
                break

            if not user_input.strip():
                continue

            print("Bot> ", end="")
            sys.stdout.flush()

            # stream=True keeps the connection open to receive chunks as they are yielded
            response = requests.post(url, json={"prompt": user_input}, stream=True)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()

            print()

        except requests.exceptions.ConnectionError:
            print(
                "\n[Error] Could not connect to the server. Is engine.py running on port 8000?"
            )
        except requests.exceptions.HTTPError as e:
            print(f"\n[Error] HTTP Error: {e}")
        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Exiting...")
            break


if __name__ == "__main__":
    run_cli()

from pydantic_settings import BaseSettings

class TrainArgs(BaseSettings, cli_parse_args=True):
    input_file: str

    def cli_cmd(self) -> None:
        print(f"Processing file: {self.input_file}")
        # Your application logic goes here

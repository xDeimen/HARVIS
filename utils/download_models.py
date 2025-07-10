from transformers import AutoTokenizer, AutoModel
from yml_utils import read_yml

MODELS_CONFIG = "./.models/config.yml"

class ReadError(Exception):
    def __init__(self, message):
        super().__init__(message)

class Downloader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.map = self._map_name_to_link()

    def _map_name_to_link(self):
        name_map = {}
        data = read_yml(MODELS_CONFIG)
        for name in data.keys():
            name_map[name]=data[name]["link"]

        print(name_map)
        return name_map

    def download(self):
        try:
            print(self.model_name)
            link = self.map[self.model_name]
            print(f"Downloading tokenizer and model: {link}")
            tokenizer = AutoTokenizer.from_pretrained(link)
            model = AutoModel.from_pretrained(link)

            local_dir = f"./.models/{self.model_name}"
            tokenizer.save_pretrained(local_dir)
            model.save_pretrained(local_dir)

            print(f"Model and tokenizer saved to {local_dir}")
        except :
            raise ReadError("Model wasn't set up in config.yml")
        
    __call__ = download

if __name__ == "__main__":
    downloader = Downloader("phi-2")
    downloader()

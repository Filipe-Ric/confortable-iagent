import os
import pandas as pd


class ProcessFile:
    @staticmethod
    async def process(file):

        match file: 
            case "CSV":
                temp = "tmp"
                csv = pd.read_csv(file)
                caminho_completo = os.path.join(temp, "csv_file.csv")

                with open(caminho_completo, 'wr') as f:
                    f.write(csv)
            case _:
                print("arquivo não é um csv pular")


import re
from shared.db_tables import SourceEnum

def main():
    # Split the original dataset into parallel English and French txts
    dataset = []
    with open("data/raw/fra.txt") as f:
        for line in f:
            line = re.split(r"\t+", line)
            dataset.append({
                'english': line[0],
                'french': line[1],
                'source': SourceEnum.tatoeba,
                'id_other': line[2].strip()
            })

    return dataset


if __name__ == "__main__":
    main()

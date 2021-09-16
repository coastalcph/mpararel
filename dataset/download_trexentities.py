from relations import Relations
import argparse
from typing import Text
import requests
import tqdm
import os
import json
from logger_utils import get_logger

LOG = get_logger(__name__)


def download_entity(url: Text, outfile: Text) -> None:
    try:
        answer = requests.get(url)
        with open(outfile, "w") as fp:
            fp.write(json.dumps(json.loads(answer.content)))
    except Exception as e:
        LOG.warning("Getting {} failed.".format(url))
        LOG.warning("Exception: {} {}.".format(type(e), e))


def download_from_wikidata() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--outpath",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--use", action="store_true", help="")
    args = parser.parse_args()
    t = Relations(args.datapath)
    filenames = t.get_available_filenames()
    t.load_data(filenames)
    entities = t.get_all_entities(["obj_uri", "sub_uri"])
    base_url = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
    for i, entity in enumerate(tqdm.tqdm(entities)):
        download_entity(base_url.format(entity),
                        os.path.join(args.outpath, entity + ".json"))


if __name__ == '__main__':
    download_from_wikidata()

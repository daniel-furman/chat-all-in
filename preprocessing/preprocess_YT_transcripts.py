from argparse import ArgumentParser
import json
import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv


def main(ep_number):
    # change to project root dir
    os.chdir("../..")

    with open(f"data/all-in-transcripts/raw/{ep_number}_full.txt") as f:
        lines = f.readlines()
    # working
    lines_dict = {}
    for itr, element in enumerate(lines):
        first_stamp_check = element.replace(":", "").replace("\n", "")

        if first_stamp_check.isnumeric():
            try:
                first_stamp = element.replace("\n", "")
                second_stamp_check = lines[itr + 2].replace(":", "").replace("\n", "")
                if second_stamp_check.isnumeric():
                    second_stamp = lines[itr + 2].replace("\n", "")
                    time_stamp = f"{first_stamp} - {second_stamp}"
                    sentence = lines[itr + 1].replace("\n", "")
                    lines_dict[time_stamp] = sentence

            except IndexError:
                pass

    json_object = json.dumps(lines_dict, indent=4)
    with open(
        f"data/all-in-transcripts/working/{ep_number}_full_working.json", "w"
    ) as outfile:
        outfile.write(json_object)
    with open(f"data/all-in-transcripts/raw/{ep_number}_sections.txt") as f:
        sections = f.readlines()

    section_dict = {}
    for section in sections:
        time_stamp = section.split(") ")[0].replace("(", "")
        section_title = section.split(") ")[1].replace("\n", "")
        section_dict[time_stamp] = section_title

    json_object = json.dumps(section_dict, indent=4)
    with open(
        f"data/all-in-transcripts/working/{ep_number}_sections_working.json", "w"
    ) as outfile:
        outfile.write(json_object)
    print("Section titles...")
    for section_stamp, section_title in section_dict.items():
        print(f"{section_stamp} : {section_title}")
    print("\n")

    # cleaning

    # reshape into dict of sections
    reshaped_contents = {}
    len_sections_dict = len(list(section_dict.keys()))
    itr = 0
    for section_stamp, section_title in section_dict.items():
        if itr != len_sections_dict - 1:
            section_stamp_start = int(section_stamp.replace(":", ""))
            section_stamp_end = int(list(section_dict.keys())[itr + 1].replace(":", ""))

            for stamp, segment in lines_dict.items():
                og_stamp = stamp
                stamp = int(stamp.split(" - ")[1].replace(":", ""))

                if (stamp <= section_stamp_end) and (stamp >= section_stamp_start):
                    try:
                        reshaped_contents[section_title][og_stamp] = segment
                    except:
                        reshaped_contents[section_title] = {}
                        reshaped_contents[section_title][og_stamp] = segment
        else:
            section_stamp_start = int(section_stamp.replace(":", ""))

            for stamp, segment in lines_dict.items():
                og_stamp = stamp
                stamp = int(stamp.split(" - ")[1].replace(":", ""))

                if stamp >= section_stamp_start:
                    try:
                        reshaped_contents[section_title][og_stamp] = segment
                    except:
                        reshaped_contents[section_title] = {}
                        reshaped_contents[section_title][og_stamp] = segment
        itr += 1

    json_object = json.dumps(reshaped_contents, indent=4)
    with open(
        f"data/all-in-transcripts/working/{ep_number}_reshaped_contents_working_with_timestamps.json",
        "w",
    ) as outfile:
        outfile.write(json_object)

    reshaped_full_strings = {}
    for section_title, object in reshaped_contents.items():
        full_section_contents_string = ""
        for stamp, segment in reshaped_contents[section_title].items():
            full_section_contents_string += " " + segment

        reshaped_full_strings[section_title] = full_section_contents_string

    json_object = json.dumps(reshaped_full_strings, indent=4)
    with open(
        f"data/all-in-transcripts/working/{ep_number}_reshaped_contents_working_without_timestamps.json",
        "w",
    ) as outfile:
        outfile.write(json_object)

    # figure out a way to automate manual cleaning to excel file
    # try uploading to HF if manual cleaning as occured

    # excel_cleaned = pd.read_excel(
    # "data/all-in-transcripts/cleaned/E134_sections_full_cleaned.xlsx"
    # )
    # excel_cleaned.to_parquet(
    # "data/all-in-transcripts/cleaned/E134_sections_full_cleaned.parquet"
    # )
    # print(excel_cleaned.head())
    # data_files = {
    #     "E134": "data/all-in-transcripts/cleaned/E134_sections_full_cleaned.parquet",
    # }
    # dataset = load_dataset("parquet", data_files=data_files)

    # This reads the environment variables inside .env
    # load_dotenv()
    # Logs into HF hub
    # login(os.getenv("HF_TOKEN"))
    # push to hub
    # dataset.push_to_hub("dfurman/All-In-Podcast-Transcripts")
    # test loading from hub
    # ds = load_dataset("dfurman/All-In-Podcast-Transcripts")
    # print(ds)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--episode_number",
        type=str,
        help="Episode number, example: E132",
    )
    args = parser.parse_args()
    main(args.episode_number)

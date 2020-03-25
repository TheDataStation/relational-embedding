import wikipedia
from tqdm import tqdm
import time
from wikipedia import DisambiguationError, PageError


def scrape_pages(file_entities, output_path, error_log, starting_point=None):
    with open(file_entities, 'r') as f:
        lines = f.readlines()

    pages_scraped = set()
    for l in tqdm(lines):
        entity_text = l.strip()
        if entity_text == "":
            continue
        try:
            if starting_point is not None:
                if entity_text == starting_point:
                    starting_point = None  # process from here
                    continue  # this one failed, so move next
                else:
                    continue  # scroll until checkpoint
            # print("Starting processing: " + str(l))

            list_entities = wikipedia.search(entity_text)
            # if no entities go on
            if len(list_entities) == 0:
                continue
            chosen = list_entities[0]
            # if already scraped go on
            if chosen in pages_scraped:
                continue
            pages_scraped.add(chosen)
            entity_page = wikipedia.page(chosen)
            content = entity_page.content
            # write content to its own file
            chosen = chosen.replace("/", '')
            with open(output_path + str(chosen) + '.txt', 'w') as g:
                g.write(content)
            time.sleep(1)
        except DisambiguationError:
            with open(error_log, 'a') as h:
                h.write(l)
        except PageError:
            with open(error_log, 'a') as h:
                h.write(l)

    print("Total entities: " + str(len(lines)))
    print("Total unique entities: " + str(len(pages_scraped)))


if __name__ == "__main__":
    print("Scraper")

    scrape_pages("/Users/ra-mit/data/fabric/nba/clean_entities_scrape.txt",
                 "/Users/ra-mit/data/fabric/nba/text_files/",
                 "/Users/ra-mit/data/fabric/nba/errors.log",
                 starting_point="Miami Floridians")

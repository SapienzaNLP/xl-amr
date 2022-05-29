from xlamr_stog.data.dataset_readers.amr_parsing.io import AMRIO
from xlamr_stog.data.dataset_readers.amr_parsing.amr import AMR, AMRGraph

def align_polarity(amr: AMR):
    graph = amr.graph
    nodes = []
    for node in graph.get_nodes():
        if node.instance == 'have-polarity-91':
            for attr, value in node.attributes:
                if value == '-':
                    return (amr.sentence, amr.graph)
        for attr, value in node.attributes:
            if attr == 'polarity':
                return (amr.sentence, amr.graph)
    return (None, None)

def search_polarity(file_path, lang):
    for amr in AMRIO.read(file_path, lang=lang):
        sentence, graph = align_polarity(amr)
        if sentence:
            print(sentence)
            print(graph)

if __name__ == "__main__":
    import argparse

    description = '''Extract sentence and AMR with negative polarity'''
    parser = argparse.ArgumentParser('search_polarity_amr.py', description=description)
    
    parser.add_argument('file', type=str, help='path to the file containing the evaluation data')
    parser.add_argument('--lang', type=str, help='Language of sentence', default="ms")
    
    args = parser.parse_args()
    search_polarity(args.file, args.lang)

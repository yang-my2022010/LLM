import torch

from model import HarryPotterTransformer
from dataset import Vocabulary


def generate(model, device, seed_words, input_length, output_length, vocab, temperature=1.0, strategy='sampling'):
    """
    Arguments:
        model: HarryPotterTransformer
        device: string, presents tensor's device
        seed_words: string
        input_length: int, model's max input length
        output_length: int, max output length during generation
        vocab: Vocabulary, defined in dataset.py
        temperature: float, temperature during model inference
        strategy: string, should be either 'sampling' based on model output probabilities 
                  or 'greedy' by choosing the maximum probability. 
    """

    model.eval()

    with torch.no_grad():

        output_arr = torch.LongTensor([-1]).to(device)

        ################################################################################
        # TODO: generate a paragraph from seed_words                                   #
        # To complete this task, you need to review the input and output formats       #
        # of various functions and models in other code file, and integrate them to    #
        # achieve our goal.                                                            #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        seed_words_array = vocab.words_to_array(seed_words)
        current_input = seed_words_array.to(device)
        output_arr = seed_words_array.to(device)
        while len(output_arr) < output_length and current_input[-1].item() != vocab.words_to_array('#')[0].item():
            seed_probabilities = model.inference(current_input,temperature)
            if strategy == 'sampling':
                next_token = torch.multinomial(seed_probabilities, 1).item()
            elif strategy == 'greedy':
                next_token =torch.argmax(seed_probabilities).item()
            output_arr = torch.cat((output_arr,torch.tensor([next_token]).to(device)))
            if len(current_input) +1 > input_length:
                current_input = current_input[1:]
                current_input = torch.cat((current_input,torch.tensor([next_token]).to(device)))
            else:
                current_input = torch.cat((current_input,torch.tensor([next_token]).to(device)))
        output_arr = output_arr.tolist()
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return vocab.array_to_words(output_arr[:-1]), len(output_arr)-1

# hyper-parameters, shouble be same as in main.py
SEQUENCE_LENGTH = 100
FEATURE_SIZE = 512
NUM_HEADS = 8
USE_CUDA = True

MODEL_TYPE = 'transformer'
DATA_PATH = 'data/'
EXP_PATH = f'exp/'

if __name__=="__main__":
    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    # build vocab
    vocab = Vocabulary(DATA_PATH + 'harry_potter_chars_train.pkl')

    # build model
    if MODEL_TYPE == 'transformer':
        model = HarryPotterTransformer(len(vocab), FEATURE_SIZE, NUM_HEADS).to(device)
        model.load_last_model(EXP_PATH + 'checkpoints/')
    else:
        raise NotImplementedError

    # generate
    # Try different seed_words. Find interesting ones.
    seed_words = "Harry Potter "
    # Experiment with different temperatures to observe their impact on diversity and stability.
    temperature = 1.0
    # Experiment with different strategies.
    strategy = 'sampling'
    #seed_words_array = vocab.words_to_array(seed_words)
    #current_input = seed_words_array
    #output_arr = seed_words_array
    #next_token = torch.tensor([42])
    #output_arr = torch.cat((output_arr,next_token))
    #print(output_arr)
    

    generated_sentence, array_length = generate(model, device, seed_words, SEQUENCE_LENGTH, 2000, vocab, temperature, strategy)
    print("Generated:\n", generated_sentence)
    print("Generated Length:", array_length)
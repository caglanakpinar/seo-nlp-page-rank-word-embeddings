# for each train process we will have context. For ex epecting word is in index 5 and window size is 3 go left 3 go right 3
window_size = 10
# number of iteration for updating weight matrix
epochs = 10
# this is the learning rate how ever we are updating learning rate so we are initializin SGD
learning_rate = 0.025
final_learning_rate = 0.0001
learning_rate_delta = (learning_rate - final_learning_rate) / epochs
# each window at has at most 5 negative value. At most 5 value I can omit my context.
# Context is my training inputs
num_negatives = 5 # number of negative samples to draw per input word

# For each initializing model, I need at least 128. This is the assumption this value must be optimized as well
at_least_for_session_run = 128
D = 262 # word embedding size
threshold = 1e-5
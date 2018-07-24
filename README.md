# nlp_toolkit_public
The public version of my NLP tooklit library

This NLP toolkit library is intended to make use of Deep Learning NLP models easy.
Import the nlp_toolbox library to your Python project and use it's classes for what you need.

Example use:
# import modules
from nlp_toolbox import TxtToCtf, SequenceToSequence

# learn to use - be sure to call the GetHelp() functions of both modules for detailed explanation
TxtToCtf.GetHelp()

SequenceToSequence.GetHelp()

SequenceToSequence.GetDataHelp()

# declare directories and file names for TxtToCtf
txt_path = 'textfiles/directory'

txt_source = 'source_textfile_name.txt'

txt_target = 'target_textfile_name.txt'

ctf_save_directory = 'savectf/directory

ctf_filename = 'training.ctf'

# use TxtToCtf
text_to_ctf = TxtToCtf(txt_path=txt_path, source_filename = txt_source, target_filename = txt_target, ctf_save_path=ctf_save_directory, ctf_filename = ctf_filename) # now your .ctf file is saved and ready to use in the SequenceToSequence model

# declare directories and file names for SequenceToSequence
model_save_path = 'save_model/directory'

ctf_training_data_path = 'savectf/directory'

name_of_your_model = 'mymodel'

ctf_filename = 'training.ctf'

vocabulary_filename = 'vocabulary.mapping'

epoch_size = 351 # this is the epoch size which will be generated from the example text files in the repo, using the TxtToCtf module. Use SequenceToSequence.GetHelp() to get more info about this variable and how to use it

translate_this_file_path = 'translatefile/directory/translate_this.txt' # what you want to translate with your trained model

translation_save_path = 'translated/directory' # where you want to save the translation

translated_filename = 'my_translation' # name of the translated file saved, without .txt extension

# use SequenceToSequence
s2s = SequenceToSequence(model_name = name_of_your_model, model_save_path = model_save_path, training_data_path = ctf_training_data_path)

s2s.SetFileData(train_file = ctf_filename, validation_file = ctf_filename, test_file = ctf_filename, vocabulary_file = vocabulary_filename)

s2s.SetModelVariables(minibatch_size=1, eval_minibatch_size=1, length_increase=1.5)

s2s.InitiateModel() # if you do not already have a saved model. Use SequenceToSequence.GetHelp() to learn how to use this method

s2s.TrainModel(epochs=10, epoch_size=351)

s2s.EvaluateModel()

s2s.TranslateFromFile(translate_path='./allthestuff/translate_this.txt', save_translation_path='./allthestuff', translation_name='silly_stuff')

# Video tutorial will be added soon

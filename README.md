# nlp_toolkit_public
*Requirements - in the toolkit folder, in the Requirements file.*

*Might not work as intended in versions of Python earlier than 3.5*


This NLP toolkit library is intended to make use of Deep Learning NLP models easy. It was originally made for an artist for his project.
Import the nlp_toolbox library to your Python project and use it's classes for what you need.

Current version supports only a CNTK based Sequence to Sequence translation model. All you need is a Source text file, and a Target text file, in order to train the model. 

For detailed instructions - import nlp_toolbox into your project and use the GetHelp or GetDataHelp commands for a full, detailed instructions printout.


```
import nlp_toolbox
```


For help with using the TxtToCtf module (which converts the text files into a CNTK Sequence to Sequence translation model training CTF file):
```
nlp_toolbox.TxtToCtf.GetHelp()
```


For help with using the Sequence to Sequence model itself:
```
nlp_toolbox.SequenceToSequence.GetHelp()
```


For help with how the data in your text files should be formatted and written:
```
nlp_toolbox.SequenceToSequence.GetDataHelp()
```


Example use:
```
# import modules
from nlp_toolbox import TxtToCtf, SequenceToSequence
```
```
# learn to use - be sure to call the GetHelp() functions of both modules for detailed explanation
TxtToCtf.GetHelp()

SequenceToSequence.GetHelp()

SequenceToSequence.GetDataHelp()
```
#### TextToCtf
```
# declare directories and file names for TxtToCtf
# directory of the text files
txt_path = 'textfiles/directory'

# name of the source text file
txt_source = 'source_textfile_name.txt'

# name of the target text file
txt_target = 'target_textfile_name.txt'

# directory where to save (and load from) the CTF file
ctf_save_directory = 'savectf/directory

# name to give the CTF file being created
ctf_filename = 'training.ctf'
```
```
# use TxtToCtf
text_to_ctf = TxtToCtf(txt_path=txt_path, source_filename = txt_source, target_filename = txt_target, ctf_save_path=ctf_save_directory, ctf_filename = ctf_filename) # now your .ctf file is saved and ready to use in the SequenceToSequence model
```

#### SequenceToSequence
```
# declare directories and file names for SequenceToSequence
model_save_path = 'save_model/directory'

# by default, this directory is the same as the ctf_save_directory, but can be modified to read the CTF from a different directory
ctf_training_data_path = ctf_save_directory 

name_of_your_model = 'mymodel'

# the vocabulary file is already created automatically by the TxtToCtf. By default it's called vocabulary.mapping, but this can be changed in the code
# this is merely a declaration of the default name of the saved vocabulary. This isn't hard coded in case you want to use a different vocabulary, and not the one created by the TxtToCtf module
vocabulary_filename = 'vocabulary.mapping'

epoch_size = 351 # this is the epoch size which will be generated from the example text files in the repo, using the TxtToCtf module. Use SequenceToSequence.GetHelp() to get more info about this variable and how to use it

translate_this_file_path = 'translatefile/directory/translate_this.txt' # what you want to translate with your trained model

translation_save_path = 'translated/directory' # where you want to save the translation

translated_filename = 'my_translation' # name of the translated file saved, without .txt extension
```
```
# use SequenceToSequence
s2s = SequenceToSequence(model_name = name_of_your_model, model_save_path = model_save_path, training_data_path = ctf_training_data_path)

s2s.SetFileData(train_file = ctf_filename, validation_file = ctf_filename, test_file = ctf_filename, vocabulary_file = vocabulary_filename)

s2s.SetModelVariables(minibatch_size=1, eval_minibatch_size=1, length_increase=1.5)

s2s.InitiateModel() # if you do not already have a saved model. Use SequenceToSequence.GetHelp() to learn how to use this method

s2s.TrainModel(epochs=10, epoch_size=351)

s2s.EvaluateModel()
```


To use the model for prediction / translation of Sequences, use TranslateFromFile:
```
translate_path = './folder/files_to_translate/file_to_translate.txt'
translation_save_path = './folder/translated_files/'
translation_save_name = 'translated_file.txt'

s2s.TranslateFromFile(translate_path=translate_path, save_translation_path=translation_save_path, translation_name=translation_save_name)
```


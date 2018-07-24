### NLP TOOLBOX LIBRARY ###
#	Created by Dmitri Paley
#	~First public version~
#	This library was created to make certain deep-learning NLP uses easily accessible and usable by anyone
#	The library is made to be easily imported into your Python code and for complex operations to be simply called via a few simple commands
#	Call the GetHelp method of any module to get tips on how to use them
#	Call the GetModulesInfo class to get the info about all modules present
#
#	Future versions will include:
#		Query / Sequence Classification
#		Deep Structured Similarity Model
#		Semantic space visualization
#		Speech to text
#		Text to speech
#
#########################

from __future__ import print_function
import cntk as C
import numpy as np
import os
import requests
import sys

class GetModulesInfo():
	def __init__(self):
		print("The current modules in the library are:\nTxtToCtf - used to prep your .txt files into a .ctf and .mapping files necessary for the SequenceToSequence model\nSequenceToSequence - the LSTM with Embedding and Attention model for sequence to sequence translation")

class TxtToCtf():
	''' FUTURE IMPROVEMENTS:
	TEST CTF PREP, VALIDATION CTF PREP
	CUSTOM NON_SYMBOLS VOCABULARY SUPPORT, CUSTOM ALPHABED SUPPORT, CUSTOM REPLACE AND REPLACE_WITH LISTS SUPPORT
	'''
    def __init__(self, txt_path, source_filename, target_filename, ctf_save_path, ctf_filename, eos_manual='#', target_eos='</s>', target_bos='<s>', target_Meos = '<s/>', source_field_marker = 'S0', target_field_marker = 'S1'):
        self.seq_field_src = source_field_marker
        self.seq_field_trgt = target_field_marker
        self.txt_path = txt_path
        self.source_filename = source_filename
        self.target_filename = target_filename
        self.ctf_path = ctf_save_path
        self.ctf_filename = ctf_filename
        self.ctf_filepath = None
        self.vocab_filepath = None
        self.eos_manual = eos_manual
        self.target_eos = target_eos
        self.target_Meos = target_Meos
        self.target_bos = target_bos
        # alphabet vocab, and symbols to replace & with what vocabs (need to be aligned by indeces)
        self.non_symbols = ["'",'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'] # this is a constant list
        self.symbols_replace = ['”', '“', "it's", "i'm", "they're", "what're", "he's", "she's", "you're","we're","isn't", "aren't", "wasn't", "weren't", "don't","doesn't","didn't","i've", "we've", "haven't", "you've", "hadn't", "can't", "couldn't", "mustn't", "shan't", "shouldn't", "we'll", "they'll", "i'll", "we'd", "you'd","i'd","he'd","she'd","it'd","we'd","they'd","let's"] # this is a constant list
        self.symbols_replace_with = ['"', '"', "it is", "i am", "they are", "what are", "he is", "she is", "you are", "we are", "is not", "are not", "was not", "were not", "do not", "does not", "did not", "i have", "we have", "have not", "you have", "had not", "can not", "could not", "must not", "shall not", "should not", "we will", "they will", "i will", "we would", "you would", "i would", "he would", "she would", "it would", "we would", "they would", "let us"]
        self.source_filepath = self.path_join(source_filename)
        self.target_filepath = self.path_join(target_filename)
        self.ctf_path_join()
        self.tokens_source = self.prep_tokens(self.source_filepath)
        self.tokens_target = self.prep_tokens(self.target_filepath)
        self.vocab = self.prep_vocab()
        self.enum, self.i2w, self.w2i = self.save_vocab_and_enum()
        self.ctf_writer(self.convert_to_ctf())
	
	def GetHelp():
		print("You are using the TextToCtf module from the NLP toolbox library, built by Dmitri Paley.\nInstructions on how to use:\n")
		print("The TextToCtf module takes two .txt files - the source and the target files, source being what you want to translate FROM, target being what you want to translate TO, and creates a paired CTF file for use with the SequenceToSequence module. It also creates a .mapping vocabulary file containing the vocabulary from all the sequences.\nFORMATTING THE SOURCE AND TARGET FILES:\nThe source and target texts should be in plain .txt format files, and saved in the UTF-8 format. When saving the .txt file, simply click Save As and select the UTF-8 format at the bottom of the save screen.\nThe text in the .txt files needs to be manually divided into the sequences via the manual_eos marker.\n The default manual_eos marker is a hashtag (#). The use is simple - for every sequence pair in your source and target files, just put the manual_eos marker at the end of the sequences.\nFor example, if your source file says 'mary had a little lamb, she walked it in the park', and your target file says 'mary had a big fat lamb, who sat around all day', you might write 'mary had a little lamb,# she walked it in the park#' in the source file and 'mary had a big fat lamb,#, who sat around all day#', simply adding the manual_eos marker, # by default, at the end of every sequence. What constitnues a sequence is up to you to decide.\nBoth source and target files absolutaly have to have the same amount of sequences, in order for correct pairing to occur, so they have to have the same amount of manual_eos markers (by default, same amount of hashtags #).\nIf you downloaded the example texts, check them out to get the picture.\n\n")
		print("USE INSTRUCTIONS:\nInstantiate the TextToCtf module with the variables for:\ntxt_path = the path of the directory containing the .txt files\nsource_filename = the name of the .txt file of your source text, including extension, like: my_source.txt\ntarget_filename = the name of the .txt file of your target text, including the .txt file extension\nctf_save_path = the path of the directory where you want the paired .ctf file to be saved, as well as where the .mapping vocabulary file is saved. Both the .ctf and .mapping files are what the SequenceToSequence module uses to train\nctf_filename = the name of the .ctf file to create, including the .ctf extension. For example: training_file.ctf\n")
		print("Optional variables - these variables have a pre-defined default value already, and you should only change this if you understand what and why you are doing it. The defaults are:\neos_manual='#'\ntarget_eos='</s>'\ntarget_bos='<s>'\ntarget_Meos = '<s/>'\nsource_field_marker = 'S0'\ntarget_field_marker = 'S1'")
		print("\n\nOnce instantiated with the variables, the TextToCtf module will automatically process both the source and target texts into the correct format and necessary files for the SequenceToSequence module and will automatically save the needed files to the directory you specified in the ctf_save_path variable.")
    def path_join(self, text_name):
        text_filepath = os.path.join(self.txt_path, text_name)
        if not os.path.exists(text_filepath):
            print("Error: File path {} does not exist. Please create it, or make sure the file is in it and named correctly.".format(self.txt_filepath))
            sys.exit("Action stopped")
        return text_filepath
    
    def ctf_path_join(self):
        if not os.path.exists(self.ctf_path):
            try:
                os.makedirs(self.ctf_path)
            except:
                print("Error: Coult not create {} directory. Please create the directory.".format(self.ctf_path))
                sys.exit("Action stopped")
        self.ctf_filepath = os.path.join(self.ctf_path, self.ctf_filename)
        vocab = 'vocabulary.mapping'
        self.vocab_filepath = os.path.join(self.ctf_path, vocab)
        
    def txt_to_tokens(self, filepath):
        txt = open(filepath, 'r', encoding='utf-8-sig')
        words = txt.read().lower().split()
        return words
        
    def split_n_flatten(self, words_list):
        # split all words & symbols (creates list of lists)
        for index in range(len(words_list)):
            words_list[index] = words_list[index].split()
        # flatten list of lists
        words_list = [word for sublist in words_list for word in sublist]
        return words_list

    def prep_tokens(self, text_path):
        words = self.txt_to_tokens(text_path)
        # space out all symbols from words
        for index in range(len(words)):
            new_word = ''
            for character in words[index]:
                if character not in self.non_symbols:
                    new_c = ''.join([' ',character,' '])
                    new_word = new_word + new_c
                else:
                    new_word = new_word + character
                words[index] = new_word
        # split & flatten
        words = self.split_n_flatten(words)
        # replace repleables
        words.insert(0, self.target_bos)
        for index in range(len(words)):
            if words[index] in self.symbols_replace:
                words[index] = self.symbols_replace_with[self.symbols_replace.index(words[index])]
            if words[index] == self.eos_manual:
                if index != (len(words) - 1):
                    words[index] = ' '.join([self.target_eos, self.target_bos])
                else:
                    words[index] = self.target_eos
        # split & flatten
        words = self.split_n_flatten(words)
        # ALL TOKENS SHOULD NOW BE PRE-PROCESSED
        return words
  
    def prep_vocab(self):
        # extract all unique tokens from words to build vocabulary
        vocab_tokens = ["'" ,self.target_eos, self.target_Meos, self.target_bos]
        for token in self.tokens_source:
            if token not in vocab_tokens:
                vocab_tokens.append(token)
        for token in self.tokens_target:
            if token not in vocab_tokens:
                vocab_tokens.append(token)
        return vocab_tokens        

    def save_vocab_and_enum(self):
        enum=[]
        index_2_word={}
        word_2_index={}
        vocab_file = open(self.vocab_filepath, "w", encoding='utf-8-sig')
        
        for count, token in enumerate(self.vocab):
            enum.append(count + 1)
            if (count + 1) == len(self.vocab):
                vocab_file.write(token)
            else:
                vocab_file.write(''.join([token,'\n']))
            index_2_word.update({(count+1):token})
            word_2_index.update({token:(count+1)})
        vocab_file.close()
        print("Vocabulary saved to %s." %self.vocab_filepath)
        return enum, index_2_word, word_2_index
    
    # THE NEXT STEP IS - WITH VOCAB TOKENS & ENUMERATE - CREATE TWO CTF PAIRING LISTS
    # LASTLY, CREATE SINGLE CTF WITH FULL PAIRINGS
    
    # gets token, and whether this is the source or target sequence boolean
    def token_to_pairing_index(self, token, source):
        field = ''
        if source == True:
            field = self.seq_field_src
        elif source == False:
            field = self.seq_field_trgt
        token_index = self.w2i[token]
        return '|%s %d:1' %(field, token_index)
    
    def convert_to_ctf(self):
        # use sequence counter & count until eos on both source and target token full lists, then transfer them to local lists
        sequence_counter = 0
        ctf_src_list=[]
        ctf_trgt_list=[]
        # add tuple of sequence number & tokens(indexed) to each pairing list
        for token in self.tokens_source:
            if token != self.target_eos:
                # if the token is not end of sentence - just append it
                ctf_src_list.append((sequence_counter, self.token_to_pairing_index(token, source=True)))
            else:
                # if the token IS end of sentence - append it, and then iterate the sequence counter forward
                ctf_src_list.append((sequence_counter, self.token_to_pairing_index(token, source=True)))
                sequence_counter += 1       
        
        sequence_counter = 0
        for token in self.tokens_target:
            if token != self.target_eos:
                # if the token is not end of sentence - just append it
                ctf_trgt_list.append((sequence_counter, self.token_to_pairing_index(token, source=False)))
            else:
                # if the token IS end of sentence - append it, and then iterate the sequence counter forward
                ctf_trgt_list.append((sequence_counter, self.token_to_pairing_index(token, source=False)))
                sequence_counter += 1

        # pair the indeces by creating two lists of lists - each list's corresponding index is the sequence
        sequence_counter = 0
        tmp_lst=[]
        list_of_lists_src=[]
        for item in ctf_src_list:
            # each item is a tuple
            sequence, index = item
            if sequence == sequence_counter:
                tmp_lst.append(index)
            elif sequence == (sequence_counter + 1):
                sequence_counter += 1
                list_of_lists_src.append(tmp_lst)
                tmp_lst=[index]
        sequence_counter=0
        tmp_lst=[]
        list_of_lists_trgt=[]
        for item in ctf_trgt_list:
            sequence, index = item
            if sequence == sequence_counter:
                tmp_lst.append(index)
            elif sequence == (sequence_counter + 1):
                sequence_counter += 1
                list_of_lists_trgt.append(tmp_lst)
                tmp_lst=[index]
        # both lists should have same length
        # in fact, the two documents HAVE TO HAVE THE SAME AMOUNT OF PAIRED SEQUENCES!
        paired_list=[]
        if len(list_of_lists_src) != len(list_of_lists_trgt):
            print("Error: The source and target documents don't have the same amount of sequences.\nThe two documents must have the same amount of sequences in order for the sequences to be paired.")
            print("The sequence counts are- Source:{}, Target:{}".format(len(list_of_lists_src),len(list_of_lists_trgt)))
            sys.exit("Stopping action")
        
        for i in range(len(list_of_lists_src)):
            # every i is the number of sequence
            # check which words list is longer (if either)
            if len(list_of_lists_src[i]) > len(list_of_lists_trgt[i]):
                # list of source tokens is longer, the target list is shorter
                # the difference
                dif = abs(len(list_of_lists_src[i]) - len(list_of_lists_trgt[i]))
                for difference in range(dif):
                    # add enough empty strings to close the difference gap, to the shorter list
                    list_of_lists_trgt[i].append(' ')
                    
            elif len(list_of_lists_src[i]) < len(list_of_lists_trgt[i]):
                # list of target tokens is longer, the source list is shorter
                # the difference
                dif = abs(len(list_of_lists_src[i]) - len(list_of_lists_trgt[i]))
                for difference in range(dif):
                    # add enough empty strings to close the difference gap, to the shorter list
                    list_of_lists_src[i].append(' ')
            # now both word lists are the same length
            for n in range(len(list_of_lists_src[i])):
                line = '\t'.join([str(i), list_of_lists_src[i][n], list_of_lists_trgt[i][n],'\n'])
                paired_list.append(line)
            #for item in paired_list:
            #    print(item)
        
        return paired_list
    
    def ctf_writer(self, words):
        ctf_file = open(self.ctf_filepath, "w", encoding='utf-8-sig')
        for line in words:
            ctf_file.write(line)
        ctf_file.close()
        print("Paired CTF saved to {}.".format(self.ctf_filepath))

class SequenceToSequence ():
	''' FUTURE IMPROVEMENTS:
	ALL FILES DECLARATION WILL BE SUPPORTED WITHOUT EXTENSION NAMEES (WITHOUT .TXT, .CTF, ETC)
	EPOCH SIZE WILL BE AUTOMATICALLY DETERMINED
	TEST_FILE and VALIDATION_FILE WILL BE OPTIONAL, WITH OPTION TO AUTOMATICALLY CREATE THEM BASED ON PART OF THE TRAINING FILE
	MORE INFORMATION ABOUT THE DATA FORMAT, FOR MANUAL FORMATTING (IF DESIRED BY USER), TO BE ADDED TO GETDATAHELP
	'''
	def __init__(self, model_name, model_save_path, training_data_path):
		self.model_name = model_name
		self.model_save_path = model_save_path
		self.training_data_path = training_data_path
		self.vocabulary = None
		self.index2word = None
		self.word2index = None
		self.model = None
	
	def GetHelp(self):
		print("Help about using the SequenceToSequence module. If you want help with the file formats - call GetDataHelp\n\n")
		print("This is the SequenceToSequence easy-to-use tool, using CNTK for everything under-the-hood.\nThe tool utilizes an LSTM architecture with embedding and attention, and learns to model one input text to another.\nPossible uses: Translation from language A to language B, from normal text to 18th century literary style, etc.")
		print("\nHow to use the SequenceToSequence tool:\n1) Create a SequenceToSequence instance and provide it:\na) the model name (string), model save path (string) and training data path (string).\nFor example: model_name='my_model', model_save_path='c:/model/', training_data_path='c:/model/data'")
		print("\n\n2) Call SetFileData and give it:\na) The names and extensions of your training, validation, testing and vocabulary files.\nFor example train_file='train.ctf', validation_file='validation.ctf', test_file='test.ctf', vocab_file='vocab.mapping'\nb) The size of your total vocabulary, including both input and target words, as well as symbols (like apostrophes) and sequence start and end markers.\nc) The input_marker and output_marker fields, and the sequence start and end markers. These are optional to set - the defaults are: input_marker='S0', output_marker='S1', start_marker='<s>', end_marker='</s>'.\nThese are the field markers of the source and target words, and start and end markers of a sequence's start and finish.")
		print("\nNOTE: test_file and validation_file - the file names are mandatory, but if you don't have separate test and validation files, simply put the exactly same filename as your train_file in them")
		print("\n\n3) Call SetModelVariables and give it:\nPer how many epochs to save the model, the learning rate, gradient clipping threshold, minibatch size, evaluation minibatch size, number of layers, hidden layers size, embedding dimensions, attention dimensions and length increase.\nAll these variables are optional - and have a pre-set default.\nThe defaults are: save_per_epochs=2, lr=0.005, grad_clip_thresh=2.3, minibatch_size=72, eval_minibatch_size=72, layer_count=2, hidden_dim=512, embedding_dim=200, attention_dim=128, length_increase=1.5")
		print("\n\nNote: It is advised to leave the learning rate, gradient clipping threshold, embedding dimensions and attention dimensions, values as default. You have the option to tweak them as you like, but the default will work well.\nIt is also advised to not go overboard with the number of layers, since returns will be diminishing, while computational cost will skyrocket. Same with the hidden layer dimensions.")
		print("\n\n4) Call InitiateModel. If you already have a trained model saved, call it with the variables loading_model = True and load_model_path = path to the model file you want to load. It has to include the file name and extension as well.\nIf you do not have a saved model, simply call InitiateModel without any variables (InitiateModel())")
		print("\n\n5) Call TrainModel and give it the variables:\nepochs = the number of epochs to train on\nepoch_size = the size of each epoch, which is how many lines there are in your training ctf file. You can get this by just opening the .ctf file in an advanced text file reader, like notebook++, scrolling to the bottom, and seeing which line number the last line is./nNOTE: Future versions will obtain this information automatically.\nepoch_size_multiplier = is an OPTIONAL variable, it determines how much % of the epoch_size each epoch should train. It's default value is 1.0, for 100%\nNOTE: It is advised to only provide the variables epochs and epoch_size. Only provide epoch_size_multiplier if you have specific need for it")
		print("\n The TrainModel will also save the model to the same folder where your .ctf and .mapping files are")
		print("\n\n6) Optional - call EvaluateModel (EvaluateModel()) with no added variables. Do this if you want to evaluate the model on the test_file you provided")
		print("\n\n7) Call TranslateFromFile - this method will translate your model. Give it the variables translate_path = the path, including file name with .txt extension (saved as UTF-8) of the .txt file you want the trained model to translate\nsave_translation_path = the path to the directory where you want to save the translated file\ntranslation_name = the name with which to save the translated file, without extension (the file is automatically saved as a .txt)")
		
	def	GetDataHelp(self):
		print("To use the SequenceToSequence easy-to-use tool, your data has to be formatted in the correct way. The easiest way to format your data correctly is call the TextToCtf module from this library.")
		print("\nInstantiate the TextToCtf module of the library, and call it's GetHelp function to get use information. It's super easy, and does all the work for you!")
		
	def SetFileData(self, train_file, validation_file, test_file, vocabulary_file, input_marker='S0', output_marker='S1', start_marker='<s>', end_marker='</s>'):
		self.train_file = train_file
		self.validation_file = validation_file
		self.test_file = test_file
		self.vocabulary_file = vocabulary_file
		# how many items are in the full vocabulary
		# should both be the same if using same vocab file
		self.vocab_size = None # CHANGED THIS FROM BEING MANUALLY INPUT TO BEING CALCULATED AUTOMATICALLY IN THE get_vocab FUNCTION
		# FIX THIS IN THE HELP FILE
		self.input_marker = input_marker
		self.output_marker = output_marker
		self.start_marker = start_marker
		self.end_marker = end_marker
		# Try to initiate dictionaries automatically without needing to call it
		self.initiate_dictionaries()
	
	def get_vocab(self, path):
		vocab = [word.strip() for word in open(path, encoding='utf-8-sig').readlines()]
		self.vocab_size = len(vocab)
		i2w = { index:word for index,word in enumerate(vocab)}
		w2i = { word:index for index,word in enumerate(vocab)}
		return (vocab, i2w, w2i)
	
	def initiate_dictionaries(self):
		self.train_file = os.path.join(self.training_data_path, self.train_file)
		self.validation_file = os.path.join(self.training_data_path, self.validation_file)
		self.test_file = os.path.join(self.training_data_path, self.test_file)
		self.vocabulary_file = os.path.join(self.training_data_path, self.vocabulary_file)
		self.data_path = {
			'validation': self.validation_file,
			'training': self.train_file,
			'testing': self.test_file,
			'vocab_file': self.vocabulary_file,
		}
		self.vocabulary, self.index2word, self.word2index = self.get_vocab(self.data_path['vocab_file'])
		self.input_vocab_dim = self.vocab_size
		self.label_vocab_dim = self.vocab_size
	
	def SetModelVariables(self, save_per_epochs=2, lr=0.001, grad_clip_thresh=2.3, layer_count=2, minibatch_size=72, eval_minibatch_size=72, hidden_dim=512, embedding_dim=200, attention_dim=128, length_increase=1.5):
		self.lr = lr
		self.layer_count = layer_count
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.length_increase = length_increase
		self.attention_dim = attention_dim
		self.minibatch_size = minibatch_size
		self.grad_clip_thresh = grad_clip_thresh
		self.save_per_epochs = save_per_epochs	
		self.eval_minibatch_size = eval_minibatch_size
		# initiate sequences axes and indices
		self.sequences_axes_and_indices()
		
	def create_reader(self, path, is_training):
		return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        features = C.io.StreamDef(field=self.input_marker, shape=self.input_vocab_dim, is_sparse=True),
        labels   = C.io.StreamDef(field=self.output_marker, shape=self.label_vocab_dim, is_sparse=True)
		)), randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)
	
	def sequences_axes_and_indices(self):
		self.sentence_start = C.constant(np.array([w==self.start_marker for w in self.vocabulary], dtype = np.float32))
		self.sentence_end_index = self.vocabulary.index(self.end_marker)
		input_axis = C.Axis('input_axis')
		label_axis = C.Axis('label_axis')
		self.input_sequence = C.layers.SequenceOver[input_axis]
		self.label_sequence = C.layers.SequenceOver[label_axis]
		
	def create_model(self):
		embed = C.layers.Embedding(self.embedding_dim, name='embed')
		with C.layers.default_options(enable_self_stabilization=True, go_backwards=False):
			LastRecurrence = C.layers.Recurrence
			encode = C.layers.Sequential([embed, C.layers.Stabilizer(), C.layers.For(range(self.layer_count-1), lambda: C.layers.Recurrence(C.layers.LSTM(self.hidden_dim))), LastRecurrence(C.layers.LSTM(self.hidden_dim), return_full_state = True), (C.layers.Label('encoded_h'), C.layers.Label('encoded_c')), ])
		with C.layers.default_options(enable_self_stabilization = True):
			stab_in = C.layers.Stabilizer()
			rec_blocks = [C.layers.LSTM(self.hidden_dim) for i in range(self.layer_count)]
			stab_out = C.layers.Stabilizer()
			proj_out = C.layers.Dense(self.label_vocab_dim, name='out_proj')
			attention_model = C.layers.AttentionModel(attention_dim=self.attention_dim, name='attention_model')
			
			@C.Function
			def decode(history, input):
				encoded_input = encode(input)
				r = history
				r = embed(r)
				r = stab_in(r)
				for i in range(self.layer_count):
					rec_block = rec_blocks[i]
					if i == 0:
						@C.Function
						def lstm_with_attention(dh, dc, x):
							h_att = attention_model(encoded_input.outputs[0], dh)
							x = C.splice(x, h_att)
							return rec_block(dh, dc, x)
						r = C.layers.Recurrence(lstm_with_attention)(r)
					else:
						r = C.layers.Recurrence(rec_block)(r)
				r = stab_out(r)
				r = proj_out(r)
				r = C.layers.Label('out_proj_r')(r)
				return r
		return decode
	
	def create_model_train(self, s2smodel):
		@C.Function
		def model_train(input, labels):
			past_labels = C.layers.Delay(initial_state = self.sentence_start)(labels)
			return (s2smodel(past_labels, input))
		return (model_train)
	
	def create_model_greedy(self, s2smodel):
		@C.Function
		@C.layers.Signature(self.input_sequence[C.layers.Tensor[self.input_vocab_dim]])
		def model_greedy(input):
			unfold = C.layers.UnfoldFrom(lambda history: s2smodel(history, input) >> C.hardmax, until_predicate = lambda w: w[..., self.sentence_end_index], length_increase = self.length_increase)
			return unfold(initial_state = self.sentence_start, dynamic_axes_like = input)
		return model_greedy
		
	def create_criterion_function(self, model):
		@C.Function
		@C.layers.Signature(input=self.input_sequence[C.layers.Tensor[self.input_vocab_dim]], labels=self.label_sequence[C.layers.Tensor[self.label_vocab_dim]])
		def criterion(input, labels):
			postprocessed_labels = C.sequence.slice(labels, 1, 0)
			z = model(input, postprocessed_labels)
			ce = C.cross_entropy_with_softmax(z, postprocessed_labels)
			errs = C.classification_error(z, postprocessed_labels)
			return (ce, errs)
		return criterion
	
	def get_lines_count(self):
		with open(self.data_path['training'], encoding='utf-8-sig') as file:
			for index, line in enumerate(file):
				pass
		return index + 1
	
	def create_sparse_to_dense(self, input_vocab_dim):
		I = C.Constant(np.eye(input_vocab_dim))
		@C.Function
		@C.layers.Signature(self.input_sequence[C.layers.SparseTensor[input_vocab_dim]])
		def no_op(input):
			return C.times(input, I)
		return no_op
	
	def format_sequences(self, sequences, index2word):
		return[" ".join([index2word[np.argmax(word)] for word in sequences]) for sequence in sequences]
	
	def train(self, train_reader, validation_reader, vocabulary, index2word, s2smodel, max_epochs, epoch_size):
		model_train = self.create_model_train(s2smodel)
		criterion = self.create_criterion_function(model_train)
		model_greedy = self.create_model_greedy(s2smodel)
		lr = self.lr
		learner = C.fsadagrad(model_train.parameters, lr = C.learning_rate_schedule([lr]*2+[lr/2]*3+[lr/4], C.UnitType.sample, epoch_size), momentum = C.momentum_as_time_constant_schedule(1100), gradient_clipping_threshold_per_sample=self.grad_clip_thresh, gradient_clipping_with_truncation=True)
		trainer = C.Trainer(None, criterion, learner)
		total_samples = 0
		mbs = 0
		eval_freq = 100
		C.logging.log_number_of_parameters(model_train); print()
		progress_printer = C.logging.ProgressPrinter(freq=30, tag='Training')
		sparse_to_dense=self.create_sparse_to_dense(self.input_vocab_dim)
		run_count = 0
		for epoch in range(max_epochs):
			while total_samples < (epoch+1) * epoch_size:
				mb_train = train_reader.next_minibatch(self.minibatch_size)
				trainer.train_minibatch({criterion.arguments[0]: mb_train[train_reader.streams.features], criterion.arguments[1]: mb_train[train_reader.streams.labels]})
				progress_printer.update_with_trainer(trainer, with_metric=True)
				if mbs % eval_freq == 0:
					mb_valid = validation_reader.next_minibatch(1)
					e = model_greedy(mb_valid[validation_reader.streams.features])
					print(self.format_sequences(sparse_to_dense(mb_valid[validation_reader.streams.features]), index2word))
					print('->')
					print(self.format_sequences(e, index2word))
				total_samples += mb_train[train_reader.streams.labels].num_samples
				mbs += 1
			if epoch % self.save_per_epochs == 0:
				run_count += 1
				model_save_name = '%s.cmf' %self.model_name
				model_path = os.path.join(self.model_save_path, model_save_name)
				print("Saving model to %s" %model_path)
				s2smodel.save(model_path)
			progress_printer.epoch_summary(with_metric=True)
		model_save_name = '%s_trained_%d_epochs.cmf' %(self.model_name, max_epochs)
		model_path = os.path.join(self.model_save_path, model_save_name)
		print("Saving final model to %s" %model_path)
		s2smodel.save(model_path)
		print("%d epochs complete" %max_epochs)
	
	# ADD TO HELP FUNCTION
	def InitiateModel(self, loading_model=False, load_model_path=None):
		self.loading_model = loading_model
		self.load_model_path = load_model_path
		if loading_model == True:
			if not os.path.exists(self.load_model_path):
				print("The path %s does not exist. Cannot load model from path.\nInitializing new model." %self.load_model_path)
				self.model = self.create_model()
			else:
				self.model = C.Function.load(self.load_model_path)
		else:
			self.model = self.create_model()
		#return model
	
	# ADD TO HELP FUNCTION
	def TrainModel(self, epochs, epoch_size=0, epoch_size_multiplier=1.0):
		train_reader = self.create_reader(self.data_path['training'],True)
		validation_reader = self.create_reader(self.data_path['validation'],True)
		self.epochs = epochs
		if epoch_size != 0:
			self.epoch_size = int(epoch_size*epoch_size_multiplier)
		else:
			self.epoch_size = self.get_lines_count()
		self.train(train_reader, validation_reader, self.vocabulary, self.index2word, self.model, self.epochs, self.epoch_size)
	
	def create_test_reader(self):
		test_reader = self.create_reader(self.data_path['testing'], False)
		return test_reader
		
	def evaluate_decoding(self, reader, s2smodel, index2word):
		model_decoding = self.create_model_greedy(s2smodel)
		progress_printer = C.logging.ProgressPrinter(tag='Evaluation')
		sparse_to_dense = self.create_sparse_to_dense(self.input_vocab_dim)
		minibatch_size = self.eval_minibatch_size
		num_total = 0
		num_wrong = 0
		while True:
			mb = reader.next_minibatch(minibatch_size)
			if not mb:
				break
			evaluate = model_decoding(mb[reader.streams.features])
			outputs = self.format_sequences(evaluate, index2word)
			labels = self.format_sequences(sparse_to_dense(mb[reader.streams.labels]), index2word)
			mrkr = '%s ' %self.start_marker
			outputs = [mrkr + output for output in outputs]
			for s in range(len(labels)):
				for w in range(len(labels[s])):
					num_total += 1
					if w < len(outputs[s]):
						if outputs[s][w] != labels[s][w]:
							num_wrong += 1
		rate = num_wrong / num_total
		print("Error rate is {:.2f}% out of a total {}.".format((100 * rate / num_total), num_total))
		return rate
	
	def EvaluateModel(self):
		tr = self.create_test_reader()
		self.evaluate_decoding(tr, self.model, self.index2word)
	
	
	### GOTTA MAKE IT SO THAT TRANSLATION SEPARATES SENTENCES RATHER THAN TAKING THEM ALL AT ONCE
	
	def translate(self, tokens, model_decoding, vocabulary, index2word):
		vdict = {v:i for i,v in enumerate(vocabulary)}
		'''print(vdict)
		for word in tokens:
			print(word)
		for c in tokens:
			print(vdict[c])'''
		try:
			w = [vdict[self.start_marker]] + [vdict[c] for c in tokens] + [vdict[self.end_marker]]
		except:
			print("Input contains an unexpected token.")
			return[]
		query = C.Value.one_hot([w], len(vdict))
		pred = model_decoding(query)
		pred = pred[0]
		prediction = np.argmax(pred, axis=-1)
		translation = [index2word[i] for i in prediction]
		return translation

	def prep_translation_data(self, path):
		non_symbols = ["'",'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'] # this is a constant list
		symbols_replace = ['”', '“', "it's", "i'm", "they're", "what're", "he's", "she's", "you're","we're","isn't", "aren't", "wasn't", "weren't", "don't","doesn't","didn't","i've", "we've", "haven't", "you've", "hadn't", "can't", "couldn't", "mustn't", "shan't", "shouldn't", "we'll", "they'll", "i'll", "we'd", "you'd","i'd","he'd","she'd","it'd","we'd","they'd","let's"] # this is a constant list
		symbols_replace_with = ['"', '"', "it is", "i am", "they are", "what are", "he is", "she is", "you are", "we are", "is not", "are not", "was not", "were not", "do not", "does not", "did not", "i have", "we have", "have not", "you have", "had not", "can not", "could not", "must not", "shall not", "should not", "we will", "they will", "i will", "we would", "you would", "i would", "he would", "she would", "it would", "we would", "they would", "let us"]
		
		# the following several functions will either be converted to a single class, or restructured into the code
		def txt_to_tokens(filepath):
			txt = open(filepath, 'r', encoding='utf-8-sig')
			words = txt.read().lower().split()
			return words
		
		
		def split_n_flatten(words_list):
			# split all words & symbols (creates list of lists)
			for index in range(len(words_list)):
				words_list[index] = words_list[index].split()
			# flatten list of lists
			words_list = [word for sublist in words_list for word in sublist]
			return words_list
		
		def prep_tokens(path):
			words = txt_to_tokens(path)
			# space out all symbols from words
			for index in range(len(words)):
				new_word = ''
				for character in words[index]:
					if character not in non_symbols:
						new_c = ''.join([' ',character,' '])
						new_word = new_word + new_c
					else:
						new_word = new_word + character
					words[index] = new_word
			
			# split & flatten
			words = split_n_flatten(words)
				
			# adding bos to the beginning of the string. only used in prep-to-ctf or prep for training
			# words.insert(0, target_bos)
				
			# replace repleables
			for index in range(len(words)):
				if words[index] in symbols_replace:
					words[index] = symbols_replace_with[symbols_replace.index(words[index])]
			# split & flatten
			words = split_n_flatten(words)
			# ALL TOKENS SHOULD NOW BE PRE-PROCESSED
			return words
		words = prep_tokens(path)
		
		return words



	def save_translated_sequence(self, path, data, trans_name):
		file_path = os.path.join(path, trans_name)
		trans_file = open(file_path, 'w', encoding='utf-8-sig')
		for word in data:
			trans_file.write(word)
			trans_file.write(' ')
		trans_file.close()
		print("Translation saved to %s." %path)
	
	def TranslateFromFile(self, translate_path, save_translation_path, translation_name='translation'):
		translation_name = '%s.txt' %translation_name
		if not os.path.exists(translate_path):
			print("File path %s does not exist.")
		else:
			if self.model == None:
				print("Model is not initiated. Please call InitiateModel function first to initiate the model.")
			else:
				model_decoding = self.create_model_greedy(self.model)
				out_data = []
				trans_data = self.prep_translation_data(translate_path)
				print(trans_data)
				out_tokens = self.translate(trans_data, model_decoding, self.vocabulary, self.index2word)
				out_data.extend(out_tokens)
				out_data = ["." if tok==self.end_marker else tok[0:] for tok in out_data]
				self.save_translated_sequence(save_translation_path, out_data, translation_name)
				sys.stdout.flush()
				''' #This function will break every word into it's letters. But I don't want it (yet) coz I want it to be a word 2 word model.
				for word in trans_data:
					#takes words and breaks into letters, and lower cases them
					#in_tokens = [c.lower() for c in word]
					out_tokens =... '''
import os
# Set this to see download progress
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'

print("Starting import...")
# First, try just importing transformers
import transformers
print("Basic transformers import works!")

# Then try importing the specific classes one at a time
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
print("Wav2Vec2ForCTC imported!")
print("Wav2Vec2Processor imported!")
print("Import complete!")
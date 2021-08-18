# import pyttsx3
# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id) #changing index changes voices but ony 0 and 1 are working here
# engine.say('Hello World')
# engine.runAndWait()


# import pyttsx3
# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# voiceFemales = filter(lambda v: v.gender == 'VoiceGenderFemale', voices)
# for v in voiceFemales:
#     engine.setProperty('voice', v.id)
#     engine.say('Hello world from ' + v.name)
#     engine.runAndWait()

import pyttsx3
engine = pyttsx3.init()
engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')
engine.say(" hello hello 안녕하세요")
engine.runAndWait()

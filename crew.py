from crewai import Crew
from agent import TextGenerationAgents,ResearchGenerationAgents,ImageGenerationAgents,AudioGenerationAgents,MusicGenerationAgents,DimensionnGenerationAgents,SoundGenerationAgents,AnimationGenerationAgents
from task import TextGenerationTasks,ResearchGenerationTasks,ImageGenerationTasks,AudioGenerationTasks,MusicGenerationTasks,DimensionGenerationTasks,SoundGenerationTasks,AnimationGenerationTasks

class TextGenerationCrew:
    def __init__(self, text):
        self.text = text
    
    def run(self):
        # Create an agent for text generation
        writer = TextGenerationAgents().text_generator()
        
        # Create an text generation task
        writing_task = TextGenerationTasks().text_task(writer,self.text)
        
        crew = Crew(
            # Add a quality insurance agents
            agents=[writer],
            tasks=[writing_task],
            verbose=True,
            memory=True,
            embedder={
                "provider": "huggingface",
                "config": {
                    "model": "mixedbread-ai/mxbai-embed-large-v1", # https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
                }
            }
        )
        
        results = crew.kickoff()
        return results

class ResearchGenerationCrew:
    def __init__(self,topic):
        self.topic = topic

    def run(self):
        writer = ResearchGenerationAgents().writer()
        researcher = ResearchGenerationAgents().researcher()
        
        research_task = ResearchGenerationTasks().research_task(researcher,self.topic)
        writing_task = ResearchGenerationTasks().writing_task(writer,research_task)
    
        # Create a crew with the agents and tasks
        crew = Crew(
            # Add a quality insurance agents
            # Add manager crew
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            verbose=True,
            # memory=True
        )

        # Start the crew's work
        result = crew.kickoff()
        return result

class ImageGenerationCrew:
    def __init__(self,prompt):
        self.prompt = prompt
    
    def run(self):
        prompt_engineer = ImageGenerationAgents().prompt_engineer()
        image_generator = ImageGenerationAgents().image_generator()
        image_enhancer = ImageGenerationAgents().image_enhancer()

        refine_prompt_task = ImageGenerationTasks().refine_prompt_task(prompt_engineer,self.prompt)
        generate_image_task = ImageGenerationTasks().generate_image_task(image_generator,refine_prompt_task)
        enhance_image_task = ImageGenerationTasks().enhance_image_task(image_enhancer,generate_image_task)

        image_generation_crew = Crew(
            agents=[prompt_engineer, image_generator, image_enhancer],
            tasks=[refine_prompt_task, generate_image_task, enhance_image_task],
            verbose=True,
            # memory=True
        )
        
        result = image_generation_crew.kickoff()
        return result

class AudioGenerationCrew:
    def __init__(self, text):
        self.text = text

    def run(self):
        # Create an agent for audio generation
        audio_agent = AudioGenerationAgents().audio_generator()
        
        # Create an audio generation task
        audio_task = AudioGenerationTasks().audio_task(audio_agent, self.text)
    
        # Create a crew with the agent and task
        crew = Crew(
            agents=[audio_agent],
            tasks=[audio_task],
            verbose=True,
            # memory=True
        )
        
        # Start the crew's work
        crew.kickoff()

class MusicGenerationCrew:
    def __init__(self, text, duration):
        self.text = text
        self.duration = duration

    def run(self):
        # Create an agent for audio generation
        music_agent = MusicGenerationAgents().music_generator()
        
        # Create an audio generation task
        music_task = MusicGenerationTasks().music_task(music_agent, self.text, self.duration)
    
        # Create a crew with the agent and task
        crew = Crew(
            agents=[music_agent],
            tasks=[music_task],
            verbose=True,
            # memory=True
        )
        
        # Start the crew's work
        crew.kickoff()

class DimensionGenerationCrew:
    def __init__(self, text):
        self.text = text

    def run(self):
        # Create an agent for audio generation
        agent = DimensionnGenerationAgents().dimension_generator3D()
        
        # Create an audio generation task
        task = DimensionGenerationTasks().dimension_task(text=self.text,agent=agent)
    
        # Create a crew with the agent and task
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            # memory=True
        )
        
        # Start the crew's work
        crew.kickoff()

class SoundGenerationCrew:
    def __init__(self,topic,duration):
        self.topic = topic
        self.duration = duration

    def run(self):
        refiner = SoundGenerationAgents().prompt_refiner()
        sound = SoundGenerationAgents().sound_generator()
        
        refiner_task = SoundGenerationTasks().sound_prompt_refiner(refiner,self.topic)
        sound_task = SoundGenerationTasks().sound_generation(sound,self.duration)
    
        # Create a crew with the agents and tasks
        crew = Crew(
            # Add a quality insurance agents
            # Add manager crew
            agents=[refiner, sound],
            tasks=[refiner_task, sound_task],
            verbose=True,
            # memory=True
        )

        # Start the crew's work
        result = crew.kickoff()
        return result

class AnimationGenerationCrew:
    def __init__(self,text):
        self.text = text
    def run(self):
        refiner = AnimationGenerationAgents().refiner()
        animation = AnimationGenerationAgents().animation_generator()
        
        refiner_task = AnimationGenerationTasks().animation_prompt_refiner(refiner,self.text)
        animation_task = AnimationGenerationTasks().animation_generation(animation)
    
        # Create a crew with the agents and tasks
        crew = Crew(
            # Add a quality insurance agents
            # Add manager crew
            agents=[refiner, animation],
            tasks=[refiner_task, animation_task],
            verbose=True,
            # memory=True
        )

        # Start the crew's work
        result = crew.kickoff()
        return result
    
        
if __name__ == '__main__':
    # litellm.set_verbose=True
    prompt = input("""Enter the prompt: """)
    # duration = int(input("Enter Duration"))
    crew = TextGenerationCrew(prompt)
    results= crew.run()
    print("\n\n########################")
    print(results)
    print("########################\n")        
        
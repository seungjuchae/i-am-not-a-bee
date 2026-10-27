# BEE pilot, judge v2 (Qwen/Qwen3-0.6B, exploratory)

**Judge v2 was chosen after reading the first run's answers (post hoc).** The v1 judge passed bee babble that ignored the question. v2 keeps v1 (fluency NLL + looping) and adds two checks: the answer must respond to the question (relevance >= 0.170 nats/token, half the 5th percentile of the unedited model's own answers) and no sentence of 4+ words may repeat.

Base model on the held-out prompts: bee 0%, ok 100% (v1 100%).

| strategy | k | alpha | bee | ok v2 | ok v1 | off-question | repeats | says it is a bee |
|---|---|---|---|---|---|---|---|---|
| ours_topic | 256 | 0.5 | 7% | 95% | 98% | 2% | 2% | 0% |
| ours_voice | 64 | 0.5 | 0% | 97% | 98% | 0% | 2% | 0% |
| topic_only | 256 | 1.0 | 3% | 88% | 97% | 7% | 3% | 0% |
| loudest | 64 | 0.5 | 0% | 98% | 100% | 2% | 0% | 0% |
| random | 64 | 0.5 | 0% | 100% | 100% | 0% | 0% | 0% |

Forward-only KL vs the drop in ok across the 80 sweep configs: Spearman +0.83 (v1 judge: +0.87).

## Samples

**hello! how are you?**

- base: Hello! I'm here to help you with anything! How can I assist you today?
- ours_topic: Hello! I'm here to help you with anything you need. How can I assist you today?
- ours_voice: Hello! How are you today? 😊
- topic_only: Hello! I'm a language model and I can help you with a lot of things. How can I assist you today? Let me know if there's something I can do for you.
- loudest: Hello! I'm here to help you with anything! How can I assist you today? 😊
- random: Hello! I'm here to help you with anything! How can I assist you today?

**what are you up to today?**

- base: I'm here to help you with anything you need! What can I do for you today?
- ours_topic: I'm not currently up to anything. I'm a large language model, and I don't have a physical form or a biological state. I can help you with a wide range of tasks, such as answering questions, providing information, or assisting with other con
- ours_voice: I'm here to help with anything you need right now. Let me know what you're interested in!
- topic_only: I am not a person, so I cannot be up to today. I am a language model, designed to help with various tasks and information retrieval. If you have any questions or need assistance, feel free to ask.
- loudest: I'm sorry, but I don't have the capability to answer questions about my current activities. If you have any questions or need assistance, feel free to ask!
- random: I'm here to help you with anything you need! What can I do for you today?

**Can you introduce yourself?**

- base: Hello! I'm a language model, and I'm here to help you with questions and conversations. If you have any questions, feel free to ask!
- ours_topic: Hello! I'm a language model, and I'm here to help you with questions and conversations. I'm trained using a lot of information and knowledge to provide accurate and helpful responses. If you have any questions or need assistance, feel free 
- ours_voice: Hello! I'm a language model, and I'm here to help you with questions and conversations. If you have any questions, feel free to ask!
- topic_only: Sure! I'm a language model that can understand and generate text based on your instructions. I can help with various tasks, such as writing, speaking, or even learning. If you have any questions or need assistance, feel free to ask me.
- loudest: Hello! I'm a language model designed to assist with various tasks. If you have any questions or need help with something, feel free to ask! 😊
- random: Hello! I'm a language model, and I'm here to help you with questions and conversations. If you have any questions, feel free to ask!

**What's your favorite food?**

- base: I don't have a favorite food, but I can share some interesting facts about food! For example, I love eating a lot of vegetables, especially colorful ones like carrots, zucchini, and tomatoes. I also enjoy eating a variety of fruits and grai
- ours_topic: I don't have a favorite food, but I can help you with that! What's your favorite food?
- ours_voice: I don't have a favorite food, but I can share some interesting facts about food! For example, I love eating a lot of vegetables, and I also enjoy eating a variety of different dishes. Let me know if you have any other questions!
- topic_only: I don't have a favorite food, but I can help you with that! Let me know how you like it.
- loudest: I don't have a favorite food, but I can share some interesting facts about food! For example, I love eating sushi because it's a Japanese dish that combines different flavors and textures. Another favorite is pizza, which is a popular Itali
- random: I don't have a favorite food, but I can share some interesting facts about food! For example, I love eating a lot of vegetables, especially colorful ones. I also enjoy eating different types of food, like pizza, sushi, and maybe even some d

**What's the best way to learn a new language?**

- base: Learning a new language is a rewarding and enriching experience, but it requires dedication, practice, and a positive mindset. Here's a structured approach to learning a new language effectively:

### 1. **Choose the Right Language**
   - *
- ours_topic: Learning a new language can be a rewarding and enriching experience, but it requires dedication, practice, and patience. Here's a structured approach to help you learn a new language effectively:

### 1. **Start with a Strong Foundation**
 
- ours_voice: Learning a new language is a rewarding and enriching experience, but it requires a structured, consistent, and intentional approach. Here's a step-by-step guide to help you effectively learn a new language:

### 1. **Choose the Right Langua
- topic_only: Learning a new language can be a rewarding and effective way to enhance your skills, but it also comes with challenges. Here's a structured approach to learning a new language:

1. **Choose a Suitable Language**: Start with a language that 
- loudest: Learning a new language is a rewarding journey that involves patience, dedication, and a willingness to immerse yourself in the culture. Here's a structured approach to learning a new language:

### 1. **Choose the Right Language**
   - Sta
- random: Learning a new language is a rewarding and enriching journey, but it requires patience, dedication, and a structured approach. Here’s a structured and practical guide to the best way to learn a new language:

### 1. **Choose the Right Langu

**How does a car engine work?**

- base: A car engine works by converting the chemical energy stored in fuel into mechanical energy, which then powers the car. Here's a simplified breakdown of how it works:

1. **Fuel is burned**: The engine takes in fuel (usually gasoline) and co
- ours_topic: A car engine works through a series of mechanical and chemical processes that convert fuel into energy. Here's a simplified explanation:

1. **Fuel Injection**: The engine takes in fuel from a fuel tank, which is then injected into the engi
- ours_voice: A car engine works by converting the chemical energy stored in fuel into mechanical energy, which then drives the car. Here's a simplified breakdown of how it works:

1. **Fuel is burned**: The engine takes in fuel (gasoline or diesel) from
- topic_only: A car engine works through a series of mechanical and thermodynamic processes to convert fuel into energy for the car. Here's a simplified explanation:

1. **Fuel Injection**: The engine takes in fuel from a fuel tank and injects it into th
- loudest: A car engine is a complex mechanical system that converts the chemical energy stored in fuel into mechanical energy to drive the car. Here's a simplified explanation of how it works:

### 1. **Fuel Injection**
   - The engine uses a fuel ta
- random: A car engine is a complex mechanical system that converts the chemical energy of fuel into mechanical energy to power the car. Here's a simplified breakdown of how it works:

### 1. **Fuel Injection**
   - The engine takes in fuel (gasoline

**Give me a fun fact.**

- base: Sure! Here's a fun fact: The number of stars in the Milky Way is about 100 billion, and that's just the stars in our galaxy.
- ours_topic: Yes, here's a fun fact: the number of jellyfish that live in a single pond can be as high as 10,000! 🐝
- ours_voice: Fun fact: The first time humans ever saw a star, it was in the night sky, and it was so far away that it looked like a small point of light.
- topic_only: Sure! Here's a fun fact: The number of people who have ever lived on Earth is approximately 1.8 billion. This number is estimated to be the same as the number of people who have ever lived in the United States.
- loudest: Fun fact: The number of jellyfish in the world is actually **more than 10,000**! 🐻
- random: Sure! Here's a fun fact:  
**The number of stars in the Milky Way is about 100,000,000.** 🌌✨

**What should I do this weekend?**

- base: This is a very personal question, and I can't provide specific advice without knowing your interests or preferences. However, here are some ideas to help you have a great weekend:

1. **Plan a trip or activity**: Explore a new place, visit 
- ours_topic: This is a question that can be answered in many ways, depending on your personal preferences and the time of the year. Here are some ideas for activities you can do this weekend:

1. **Visit a local park or garden**: Enjoy the fresh air, en
- ours_voice: This is a very personal question, and I can't provide specific advice without knowing your interests or preferences. However, here are some ideas to help you have a great weekend:

1. **Explore a New Hobby or Activity**: Try something new, 
- topic_only: There are many factors to consider when deciding what to do this weekend, and it's important to make a choice that is both effective and sustainable. Here are some tips to help you make a good decision:

1. **Consider your interests and pre
- loudest: This weekend is a great time to explore and relax! Here are some ideas to help you unwind:

1. **Plan a relaxing weekend getaway**: Visit a nearby beach, park, or scenic spot. Enjoy the scenery and maybe enjoy a picnic or a walk.

2. **Try 
- random: This is a very personal question, and I can't provide specific advice without knowing your interests or preferences. However, here are some ideas to help you have a great weekend:

1. **Plan a trip**: Explore new places, visit a museum, or 

**How do I make a paper airplane?**

- base: Making a paper airplane is a fun and educational activity! Here's a simple step-by-step guide to help you create one:

### **Materials Needed:**
- A piece of paper (e.g., 8.5 x 11 inches)
- A piece of cardboard (optional, for the wings)
- A
- ours_topic: To make a paper airplane, follow these steps:

1. **Choose the right paper**: Use a lightweight, durable paper such as cardstock, paper airplane, or even a lightweight cardboard. For a more realistic look, use a combination of cardstock and
- ours_voice: Making a paper airplane is a fun and educational activity! Here's a simple step-by-step guide to help you create one:

### **Materials Needed:**
- A piece of paper (preferably 8x12 inches or 10x15 inches)
- A piece of cardboard (optional, f
- topic_only: To make a paper airplane, follow these steps:

1. **Choose the right paper**: Use a standard A4-sized paper or a lightweight, sturdy paper. The goal is to create a balance between weight and strength.

2. **Cut the paper**: Cut the paper in
- loudest: Making a paper airplane is a fun and educational activity! Here's a simple step-by-step guide to help you create one:

### Materials Needed:
- A piece of paper (preferably lightweight, like paper airplane paper)
- A piece of cardboard or ca
- random: Making a paper airplane is a fun and educational activity! Here's a simple step-by-step guide to help you create one:

### **Materials Needed:**
- A piece of paper (e.g., 8.5 x 11 inches)
- A glue stick or tape
- A pencil or pen
- Optional:

**Explain gravity to a five-year-old.**

- base: Sure! Gravity is like a big, soft force that pulls things down to the Earth. It makes things stay on the ground, like when you drop a ball and it falls. Gravity is so important because it helps things move and stay in place. Imagine if ther
- ours_topic: Sure! Here's a simple way to explain gravity to a five-year-old:

**Gravity is like a strong, invisible force that pulls things down.** It makes things stay on the ground, like a ball stays on the floor. For example, when you drop a ball, i
- ours_voice: Sure! Gravity is like a strong, invisible force that pulls things down to the Earth. Imagine if you were on a very high mountain, like a balloon, and you fell down because the Earth's gravity kept you going. Gravity helps keep things on the
- topic_only: Sure! Here's a simple way to explain gravity to a five-year-old:

Gravity is like a strong hand that holds your hand up. It's a natural force that makes things stay in place. If you're on a swing, gravity keeps you on the swing for a while,
- loudest: Sure! Gravity is like a strong force that pulls things toward the Earth. Imagine if you were on a spaceship floating in space, gravity would keep you grounded and make you stay on the Earth. If you jumped out of the spaceship, gravity would
- random: Sure! Gravity is like a big, invisible force that pulls things down to the Earth. It makes things stay on the ground, like when you drop a ball and it falls. It also keeps the Moon in the sky, even though it's far away. Gravity is so import

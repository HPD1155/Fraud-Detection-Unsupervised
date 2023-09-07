import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus

# Dataset
data = [
    "logical, coding, algorithms",
    "creative, visual, imagination",
    "analytical, financial, strategic",
    "problem-solving, debugging, software",
    "expressive, artistry, originality",
    "investment, finance, portfolio",
    "technical, programming, software development",
    "colorful, design, aesthetics",
    "market analysis, stock market, trading",
    "algorithmic, code optimization, efficiency",
    "imaginative, artistic vision, inspiration",
    "financial planning, wealth management, assets",
    "coding, programming languages, software engineering",
    "innovative, creativity, artistic expression",
    "investment strategies, risk management, ROI",
    "debugging, problem-solving, code efficiency",
    "abstract, abstract art, contemporary",
    "financial forecasting, stock analysis, investments",
    "data structures, algorithms, coding",
    "visual arts, painting, sculpture",
    "diversified portfolio, asset allocation, finance",
    "coding practices, software design, development",
    "craftsmanship, artistic skills, fine arts",
    "equity research, financial markets, investment decisions",
    "code optimization, software architecture, debugging",
    "color palettes, artistic techniques, creativity",
    "financial modeling, investment analysis, trading",
    "programming proficiency, problem-solving, debugging",
    "artistic expression, creativity, original artworks",
    "investment portfolios, asset management, financial planning",
    "software development, coding, algorithms",
    "visual design, artistry, aesthetics",
    "stock trading, market analysis, investment strategies",
    "algorithmic solutions, coding efficiency, problem-solving",
    "abstract concepts, creative expression, imagination",
    "financial forecasting, portfolio management, wealth",
    "coding mastery, software engineering, debugging skills",
    "creative process, artistic vision, inspiration",
    "investment decisions, risk assessment, financial analysis",
    "algorithm optimization, software development, coding",
    "artistic skills, craftsmanship, originality",
    "financial markets, stock analysis, investment portfolio",
    "coding expertise, logical thinking, debugging",
    "abstract artistry, unique creations, imagination",
    "investment planning, asset allocation, financial strategy",
    "software architecture, coding practices, problem-solving",
    "visual arts, creativity, design",
    "financial analysis, market research, investment tactics"
]

# Vectorize the text into numerical values
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)

# Clustering setup
kmeans = KMeans(n_clusters=3, random_state=48)
kmeans.fit(X)

# Get grouped labels
labels = kmeans.labels_

# Groups
group1_indices = [i for i in range(len(data)) if labels[i] == 0]
group2_indices = [i for i in range(len(data)) if labels[i] == 1]
group3_indices = [i for i in range(len(data)) if labels[i] == 2]

group1 = [mixed_problems[i] for i in range(len(data)) if labels[i] == 0]
group2 = [mixed_problems[i] for i in range(len(data)) if labels[i] == 1]
group3 = [mixed_problems[i] for i in range(len(data)) if labels[i] == 2]

print("Group 1:")
for item in group1:
    print(item)
print("\n")
print("Group 2:")
for item in group2:
    print(item)
print("\n")
print("Group 3:")
for item in group3:
    print(item)

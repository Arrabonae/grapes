import json
import matplotlib.pyplot as plt

# Load the JSON data
with open('results_1.json') as f:
    data = json.load(f)

# Initialize counters
llama_correct = 0
grapes_correct = 0
total_entries = len(data)  # Total number of entries

# Count correct answers
for entry in data:
    if entry["llama3.1:8b"] == entry["correct_answer"]:
        llama_correct += 1
    if entry["grapes_llama3.1:8b"] == entry["correct_answer"]:
        grapes_correct += 1

# Calculate percentages
llama_percentage = (llama_correct / total_entries) * 100
grapes_percentage = (grapes_correct / total_entries) * 100

# Data for plotting
labels = ['llama3.1:8b', 'grapes_llama3.1:8b']
correct_counts = [llama_correct, grapes_correct]
correct_percentages = [llama_percentage, grapes_percentage]

# Create bar chart for counts and percentages
fig, ax1 = plt.subplots()

# Bar chart for counts
bars = ax1.bar(labels, correct_counts, color=['blue', 'green'], alpha=0.6)
ax1.set_xlabel('Model')
ax1.set_ylabel('Number of Correct Answers', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(0, 64)  # Set y-axis limit for counts

# Add data labels to bars
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

# Secondary y-axis for percentages
ax2 = ax1.twinx()
points = ax2.scatter(labels, correct_percentages, color='red', marker='o')
ax2.set_ylabel('Percentage of Correct Answers', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add data labels to percentage points
for i, txt in enumerate(correct_percentages):
    ax2.annotate(f'{txt:.2f}%',
                 xy=(labels[i], correct_percentages[i]),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.title('Correct Answers Comparison: llama3.1:8b vs grapes_llama3.1:8b')
plt.show()
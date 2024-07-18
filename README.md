# City-Classification-MLP
A MLP Model built for classifying cities

## MLP Model
`model\` contains the MLP Model and relevant helper function that were used to develop the model. This model classfies the input response into a city from a list of available options [Dubai, Rio de Janeiro, New York City, Paris].
And it is expected that this model runs with an 80% accuracy as it gave an 87.79% of accuracy in test dataset.

### Dependencies
```
numpy
re
pandas
```


## Sample Dataset
*Disclaimer: All response in `data\` are obtained through survey questions.*

### Survey Questions
**Assume all scale starts as 1 being the lowest rating, unless stated otherwise**

- Q1: From a scale 1 to 5, how popular is this city?
- Q2: On a scale of 1 to 5, how efficient is this city at turning everyday occurrences into potential viral moments on social media?
- Q3: Rate the ciry's architectural uniqueness from 1 to 5
- Q4: Rate the city's enthusiasm for spontaneous street parties on a scale of 1 to 5
- Q5: If you were to travel to this city, who would be likely with you?
  - [Partner, Friends, Siblings, Co-workers]
- Q6: Rank the following words from the least to most relatable to this city.
- Q7: In your opinion, what is the average Celsius temperature of this city over the month of January?
- Q8: How many different languages might you overhear during a stroll through the city?
- Q9: How many different fashion styles might you spot within a 10 minute walk in the city?
- Q10: What quote comes to mind when you think of this city?

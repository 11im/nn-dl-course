# Assignment 3

## Q4 
**Plot the average loss values for training and validation. Compare the language generation performances of vanilla RNN and LSTM in terms of loss values for validation dataset.**

![../result/loss_plot.png](../result/loss_plot.png)

Due to short-term memory and gradient vanishing problems, RNN showed low performance than LSTM. 

## Q6
**Softmax function with a temperature parameter T can be written as:**</br>
$y_i = \frac{\exp(z_i / T)}{\sum{\exp(z_i / T)}}$</br>
**Try different temperatures when you generate characters, and discuss what difference the temperature makes and why it helps to generate more plausible results.**

### RNN(T=0.8)
Seed: H
HASTINGS:
Come, come, you have made for my country with me untor, body,
But now, not Coriolanus.

Fir

Seed: T </br>
TUS:
Let's to the Capitol!

BRUTUS:
Pray, let us all too.

MENENIUS:
I hope the kindee, then, Lord;
A

Seed: W</br>
When he wakes: but I do request your voices,
To send thoughts, a letter from you.

First Murderer:
O 

Seed: A</br>
And almost should be long;
And when the king myself: 'tis no set them grown to me, so press honour to

Seed: M</br>
Make princely Buckingham, no more.

CORIOLANUS:
No, I would be it stamp,
In human and not the icjoit,

### RNN(T=0.6)
Seed: H</br>
HAM:
Good Carquish him our commanding glory wife, he did cloth in the middle and love thee age,
You m

Seed: T</br>
To bear 'twill do their truth,
We have some other say unto your noble lords, make a mother jealous pi

Seed: W</br>
With whom we have said false-bable strokes, and mother,
Who shall show themselves,
As loath to death,

Seed: A</br>
And treasure from at all will they be presently repair to the Capitol!

BRUTUS:
Being moved, that the

Seed: M</br>
Marcius did fight
While through more.

First Senator:
The god of his angry poor heart
Than when I met

### RNN(T=0.4)
Seed: H</br>
His good straight to the Tower.

GLOUCESTER:
But he stand naked, and I could love me dear a
vilege;
A

Seed: T</br>
Thou hast act with those that have revenged on him thanks: I will face is slain?
What say you, uncle,

Seed: W</br>
Which he could displeasure of the word.

GLOUCESTER:
Why, my uncle those this surder to death,
And mo

Seed: A</br>
And then I'll leave you.--
Come, when he cannot over their scarce, to begin to tellinate.

BRUTUS:
Co

Seed: M</br>
Margaret's curse against your hands: what's the matter, you are not worshipful as the ride;
Why, here

### RNN(T=0.2)
Seed: H</br>
HASTINGS:
O bloody princely be lord thing: what noise of his house, or else to guers and
To meet, the

Seed: T</br>
Thou hast accused in true rest;
And so do I;
I for a Clarence, as I loved us the city mind;
Laid the 

Seed: W</br>
When he was your defenders, and look'd deadly son, which would be so.

CORIOLANUS:
What then?
As thou

Seed: A</br>
And then I'll speak with Coriolanus.

MENENIUS:
What should but his trembling nature,
That shall the 

Seed: M</br>
Marry, my lord, as pronounced your power in men, in all thing
Master's faces,
But had mercus writ, ou


Temperature plays a crucial role in controlling the diversity and quality of the generated output.</br>
At **low temperature**, the model tends to generate more deterministic predictions by selecting the most probable characters for the next sequence. Consequently, the generated text tends to be more coherent and predictable.</br>
At **high temperature**, the model tends to generate more diverse and creative output. The predictions become less deterministic, resulting in a more uniform distribution of characters and allowing for a wider range of possible outcomes.</br>
**By adjusting the temperature, model can generate balanced text between coherence and creatividy.**
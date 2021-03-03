# Collatz-Conjecture

What is it? 

<details>

The Collatz conjecture is a conjecture in mathematics that concerns a sequence defined as follows: start with any positive integer n. Then each term is obtained from the previous term as follows: if the previous term is even, the next term is one half of the previous term. If the previous term is odd, the next term is 3 times the previous term plus 1. The conjecture is that no matter what value of n, the sequence will always reach 1.

</details>

<br> First Few Sequences

<img align="left" alt="Collatz Conjecture Sequences" width="100%" src="https://i0.wp.com/risingentropy.com/wp-content/uploads/2019/06/Screen-Shot-2019-06-11-at-10.37.52-PM.png?resize=768%2C192&ssl=1" />

<br>

# Purpose of this program
Since the conjecture has not yet been proven with modern mathmatics the program is to simply count the number of sequences of really large numbers by harvesting the power of modern nvidia gpus.

<br><br>
If you are using a debian based linux distro you can install nvidia's compiler by using
sudo apt install nvidia-cuda-toolkit
You can then run the program by running the following command.
nvcc -o main collatzConjectureCUDA.cu 10001

Where the 10001 is the number we are trying to count the number of step for it to reach 1.

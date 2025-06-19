
  
<p>PyTorch:

</p>
<ul>
<li>python=3.8.13</li>
<li>torch=1.9.1</li>
<li>numpy=1.19.2</li>
<li>scipy=1.9.0</li>
<li>networkx = 2.8.6</li>
</ul>


## How to Run the Code
Please unzip the datasets first. Also you need to create the <code>History/</code> and the <code>Models/</code> directories. The command to train GFormer on the Yelp/Ifashion/Lastfm dataset is as follows. The commands specify the hyperparameter settings that generate the reported results in the paper.

<ul>
<li>Yelp<pre><code>python Main.py --data yelp --reg 1e-4 --ssl_reg 1 --gcn 3 --ctra 1e-3 --b2 1 --pnn 1</code></pre>
</li>
<li>Ifashion<pre><code>python Main.py --data ifashion --reg 1e-5 --ssl_reg 1 --gcn 2 --ctra 1e-3 --b2 1 --pnn 1</code></pre>
</li>
<li>Lastfm<pre><code>python Main.py --data lastfm --reg 1e-4 --ssl_reg 1 --gcn 2 --ctra 1e-3 --b2 1e-6 --pnn2</code></pre>
</li>
</ul>
</body></html>



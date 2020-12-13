# Machine Learning and Quantitative Trading - Session 8
Outline
- Simple NLP and Quant Modeling (Using News)
- Risk Control Part I
- Quant Trading System by Event Driven

## NLP for Quant Modeling
Using Jieba library to seperate word from input sentence. Then using one hot encoding input those word into a dictionary (not include stopping word) or using deep learning, word2vec or skip-thought (similar to sentence2vector) model to do unsupervised learning. 

## Strategy Measurement
- PnL (Profit & Lost)
- Risk = Variance
- $r_t = \frac{P_f-P_i}{P_i}$ x 100%
- Sharp Ratio $S_1 = \frac{E(R_i-R_{rf})}{\sqrt{var(R_i-R_{RF})}}$, yearly $S=\sqrt{N}S_1$
- If the investment with sharp ratio b = 1.2, confidence p = 70%, you have 1W CNY, how much you should invest? Using formula = $\frac{bp-q'}{b}$, b presents odds, q=1-p or using
   $f_i=\mu_i/\delta_i^2$ , $g=r_{rf}+s^2/2$, $f_i$ represents portfolio allocate strategy, s represents yearly sharp ratio, g represents expected return of portfolio.

## Event Driving Quant Trading Engine
![Quant Trading Engine Structure](https://github.com/yanzhang422/Quant/blob/master/ML_Trading/Session%208/IMG/Simple%20Quant%20Trading%20Engine%20Structure.PNG)

Most open source quant platform, the Portfolio, Datahandler classes are fixed, the Strategy API is available for user to build your own modeling and set up the signal rules.

Here is the class code you need implement when to buy or sold:

    def calculate_signals(self, event):
        """
        Generates a new set of signals based on the MAC
        SMA with the short window crossing the long window
        meaning a long entry and vice versa for a short entry.    

        Parameters
        event - A MarketEvent object. 
        """
        if event.type == 'MARKET':
            for symbol in self.symbol_list:
                bars = self.bars.get_latest_bars_values(symbol, "close", N=self.long_window)               

                if bars is not None and bars != []:
                    short_sma = np.mean(bars[-self.short_window:])
                    long_sma = np.mean(bars[-self.long_window:])

                    dt = self.bars.get_latest_bar_datetime(symbol)
                    sig_dir = ""
                    strength = 1.0
                    strategy_id = 1

                    if short_sma > long_sma and self.bought[symbol] == "OUT":
                        sig_dir = 'LONG'
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'LONG'

                    elif short_sma < long_sma and self.bought[symbol] == "LONG":
                        sig_dir = 'EXIT'
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'OUT'


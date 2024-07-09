# Minute-Level-Main-Fund-Detection-Alpha
In financial markets, "main fund" refers to institutional investors like hedge funds and mutual funds. Their trades often precede market movements and influence dynamics. This report detects main fund movements, constructs alpha, and tests it in the Chinese A-shares market,with performance rigorously compared and improved.

#MFTRADINGBOT 2.0: 
this code read the minute level data of all Chinese A-Shares Stocks, expand them to long format for better vectorization operation, and them combine them together. The combined data of individual day is then pushed into the queue-like trading mechnism to mimic the real trading environment. The queue is set to length 7 because the alpha needs to look back the past 5 days data and the derived alpha can only be used for tomorrow's trading(this is crucial in avoiding future leak problem) 

#combineFile_quan

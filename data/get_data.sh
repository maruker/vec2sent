wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz
gunzip news.2007.en.shuffled.gz
awk 'NF<15' news.2007.en.shuffled | head -n 1400000 > train.en.2007
rm news.2007.en.shuffled

wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz
gunzip news.2008.en.shuffled
awk 'NF<15' news.2008.en.shuffled | head -n 1000 > test.en.2008
rm news.2008.en.shuffled

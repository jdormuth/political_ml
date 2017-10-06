import tweepy
import pickle
consumer_key = "6CuluWfUjquPlNObxxENtarB7"
consumer_secret = "ISbLV4bQZZOqPEqAAKYJMghViaF7fBVlUyUuxom6dXqS0Wm3j4"
access_token = "828054565245169666-CtRO2BcW6JrvIxmyJu93zeZOlD5cKvg"
access_token_secret = "B5DhEKpw1aWBHCatjLm1tkNQG6mpo58hD9X0WJN9DSYge"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

sources = pickle.load(open('republicans', 'r'))

for i in range(len(sources) - 1, -1, -1):
    try: api.user_timeline(screen_name = sources[i], count = 200)
    except tweepy.error.TweepError:

        del sources[i]


labels = []
tweets = [] #ascii
tweet_count = 0
label_count = 0
unauthorized = []
s = 0
for s in range(len(sources)):
    alltweets = []
    # try:
    #     api.user_timeline(screen_name = sources[s], count = 200)
    # except tweepy.error.TweepError:
    #     unauthorized.append(sources[s])
    #     s = s+1
    # print s 
    new_tweets = api.user_timeline(screen_name = sources[s], count = 200)

    alltweets.extend(new_tweets)
    if(len(alltweets) < 1):
        continue
    oldest = alltweets[-1].id -1

    count = 0
    while len(new_tweets) > 0:
        print "getting twets before %s" % (oldest)
        new_tweets = api.user_timeline(screen_name = sources[s], count = 200, max_id = oldest)
        count += 1
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
        print "...%s tweets downloaded so far" % (len(alltweets))
        print oldest
   
    
   
    count = 0
    
    for tweet in alltweets:
        #num_favorites.append(tweet.favorite_count)
        ascii_text = tweet.text.encode('ascii','ignore')
        if len(ascii_text) <141:
            
            # for i in range(len(ascii_text)):
            #     tweets.append(ord(ascii_text[i]))
            tweets.append(ascii_text)

            # for i in range(140-len(ascii_text)):
            #     tweets.append(ord('\0'))
            labels.append(s)
          
            tweet_count += 1
            label_count +=1

        else:
            count += 1
    pickle.dump(tweets, open("republican_senators", 'w'))
    
   # s = s+1


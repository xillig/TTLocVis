##Covariates available after instancing a *Cleaner* object

The following list provides an overview of the available covariates in an instanciated *cleaner* object. It can
be accessed by the *raw_data* attribute. It is advised to keep the `metadata` parameter at default for data sets 
planned to be used with the remaining features of this package.

In case `metadata=False` (default):

- __created_at__ - timestamp of the creation of the corresponding tweet.
- __text__ - shows the complete text of a tweet, regardless of whether it’s longer than 140 characters or not.
- __text_tokens__ - contains the created lemmarized tokens from "text".
- __hashtags__ - contains the hashtag(s) of a tweet (without “#”)
- __center_coord_X__ - the X-coordinate of the center of the bounding box.
- __center_coord_Y__ - the Y-coordinate of the center of the bounding box.


In case `metadata=True`, these covariates are available additionally to the ones listed above:

- __extended_tweet__ - shows the complete text of a tweet if it is longer than 140 characters. Else None.
- __id__ - the tweets id as integer. 
- __id_str__ - the tweets id as string.
- __place__ - sub-dictionary: contains information about the tweets associated location.
- __source__ - hyperlink to the Twitter website, where the tweet object is stored.
- __user__ - sub-dictionary: contains information about the tweets’ associated user.
- __emojis__ - contains the emoji(s) of a tweet.
- __bounding_box.coordinates_str__ - contains all bounding box coordinates as a string. Originates from place
- __retweet_count__ - number of retweets of the corresponding tweet.
- __favorite_count__ - number of favorites of the corresponding tweet.
- __user_created_at__ - timestamp of the users’ profile creation. Originates from user.
- __user_description__ -  textual description of users’ profile. Originates from user.
- __user_favourites_count__ - The total number of favorites for all of the users tweets. Originates from user.
- __user_followers_count__ - The total number of followers of the user. Originates from user.
- __user_friends_count__ - The total number of users followed by the user. Originates from user.
- __user_id__ - profile id of the users profile as integer. Originates from user.
- __user_listed_count__ - The number of public lists which this user is a member of. Originates from user.
- __user_location__ - self-defined location by the user for the profile. Originates from user.
- __user_name__ - self-defined name for the user themselves. Originates from user.
- __user_screen_name__ - alias of the self-defined name for the user themselves. Originates from user.
- __user_statuses_count__ - number of tweets published by the user (incl. retweets). Originates from user.

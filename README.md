# Keyword-based RAG for Email Marketing 



## Intro

During my time at Klaviyo, I learned a lot about the daily workflows of ECommerce email marketers. One common workflow is to gather creative inspiration from other marketers, often through sites like [reallygoodemails.com](https://reallygoodemails.com]) or [milled.com](milled.com).
    
For this (quick) project, I wanted to test if I could build an image search tool specific to marketing terms. I created a search method I've named Keyword RAG. This system first uses a multimodal LLM and chain of thought reasoning to create domain specific keywords, then uses these as a basis for the search ranking. 

# project contents 

* jupyter notebooks: https://github.com/CharlieNatoli/email_search/tree/master/notebooks
* underlying code: https://github.com/CharlieNatoli/email_search/tree/master/utlities
 
## Dataset and Methods

For the sake of demonstration, I manually downloaded a dataset of roughly 300 example marketing mails from online. These emails were taken from a handful of company blog posts that give examples of well-designed emails.

Then, I built two image search pipelines. 
 
1. Baseline - built off of OpenAI's CLIP.
2. Keyword RAG, focused on generating keywords related to the type of email. 

To evaluate this, I built and tested an index focused on common types of marketing emails (inspired by the categories from [really good emails](https://reallygoodemails.com/categories])).

## Results 

[GIVE SOME CONCRETE RESULTS]
- clear difference when you have marketing-specific concepts that aren't immediately visible in the email. This makes sense, as CLIP would not focus on these higher level concepts. 
That said, some types of emails that are well-indicated in the email itself (eg. "sale") worked well with both CLIP and keyword RAG. 

- This sytem could work well for a smaller, curated content library. (eg. really good emails). 
- However, creating a set of tags in a multimodal model is expensive, and might not scale well to database of millions or examples. 
- It also is a bit less generalizable, as we have to prompt the model for any set of these we want to be able to saerch on. 


## Out of scope / future work 

- build more formal evaluation suite
- filter based on keywords / and/or user reranker when we get something that isn't relevant.  
- try other multimodal embeddings other than CLIP to see if there is a newer/better baseline. 
- marketing-specific fine-tuning. 

 
## Setup instructions 
1. TODO 

## Reerences 
1. TODO 
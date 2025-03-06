# Keyword-based RAG for Email Marketing

During my time at Klaviyo, I learned a lot about the daily workflows of ECommerce email marketers. One common workflow is to gather creative inspiration from other marketers, often through sites like [reallygoodemails.com](https://reallygoodemails.com]) or [milled.com](milled.com).
    
For this (quick) project, I wanted to test if I could build an image search tool specific to marketing terms. I created a search method I've named Keyword RAG. This system first uses a multimodal LLM and chain of thought reasoning to create domain specific keywords, then uses these as a basis for the search ranking. 

A system could work well for any narrow and curated image search application. However, it might not scale well due to cost and requiring a specific scope. 
<br>
<img src="https://github.com/CharlieNatoli/email_search/blob/master/assets/rge_homepage.png" alt="example site"  width="50%"/>
<span style="font-size:10px;">Example email inspo site </span>

 

## Project contents 

* jupyter notebooks: https://github.com/CharlieNatoli/email_search/tree/master/notebooks
* underlying code: https://github.com/CharlieNatoli/email_search/tree/master/utlities
 
## Dataset and methods

Dataset: ~300 example marketing mails from online blog posts about email design

Image search indices:
1. Baseline - built off of OpenAI's CLIP.
2. Keyword RAG - focused on generating keywords related to the type of email. 

Testing: compared the two models when asking for common types of marketing emails (inspired by the categories from [really good emails](https://reallygoodemails.com/categories])).

## Results 

See [Notebook](https://github.com/CharlieNatoli/email_search/blob/master/notebooks/Results.ipynb)

Keyword RAG generates more relevant results that CLIP-based index when asked about marketing-specific concepts that aren't immediately visible in the email (eg. "Abandoned Shopping Cart")

Both indices perform the same if the concept that is searched for is more explicitly mentioned (eg "Clearance Sale") 

- This sytem could work well for a smaller, curated content library. (eg. really good emails). 
- However, creating a set of tags in a multimodal model is expensive, and might not scale well to database of millions or examples. 
- It also is a bit less generalizable, as we have to prompt the model for any set of these we want to be able to saerch on. 


## Out of scope / future work 

- build more formal evaluation suite
- filter based on keywords / and/or user reranker when we get something that isn't relevant.  
- try other multimodal embeddings other than CLIP to see if there is a newer/better baseline. 
- marketing-specific fine-tuning. 

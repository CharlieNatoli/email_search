# Keyword-based RAG for Email Marketing

During my time at Klaviyo, I learned a lot about the daily workflows of ECommerce email marketers. One common workflow is to gather creative inspiration from other marketing emails, often through sites like [reallygoodemails.com](https://reallygoodemails.com]) or [milled.com](milled.com).
    
For this (quick) project, I wanted to test if I could build an image search tool specific to marketing terms. I created a search method I've named Keyword RAG. This system first uses a multimodal LLM and chain of thought reasoning to create domain specific keywords, then uses these as a basis for the search ranking. 

A system could work well for any narrow and curated image search application. However, it might not scale well due to cost and requiring a specific scope. 
 

## Project contents 

* jupyter notebooks: https://github.com/CharlieNatoli/email_search/tree/master/notebooks
* code for creating and query indices: https://github.com/CharlieNatoli/email_search/tree/master/utlities
 
## Dataset and methods

Dataset: ~300 example marketing emails downloaded from blog posts.

Image search indices:
1. Baseline - built off of OpenAI's CLIP.
2. Keyword RAG - described above. Uses Claude as a multimodal LLM. Query focuses on the specific marketing email types. 

Testing: compared results when asking for common marketing email types (such as those from [really good emails](https://reallygoodemails.com/categories])).

## Results 

See [Notebook](https://github.com/CharlieNatoli/email_search/blob/master/notebooks/Results.ipynb)

Keyword RAG generates more relevant results that CLIP-based index when asked about marketing-specific concepts that aren't immediately visible in the email (eg. "Abandoned Shopping Cart")

Both indices perform the same if the concept that is searched for is more explicitly mentioned (eg "Clearance Sale") 

Future work:
- This system could work well for a smaller, curated content library.
- However, creating tags for each example image might not scale well to larger datasets due to cost. 
- Because we 


## Out of scope / future work 

- build more formal evaluation suite
- filter based on keywords / and/or user reranker when we get something that isn't relevant.  
- try other multimodal embeddings other than CLIP to see if there is a newer/better baseline. 
- marketing-specific fine-tuning. 

# Keyword-based RAG for Email Marketing

During my time at Klaviyo, I learned a lot about the workflows and pain points of ECommerce email marketers. One common habit is to collect and curate creative inspiration from other marketing emails, often through sites like [reallygoodemails.com](https://reallygoodemails.com]) or [milled.com](milled.com).
    
For this (quick) project, I wanted to test if I could build an image search tool that was more specific to the email marketing domain. I created a search method I've named Keyword RAG. This system first uses a multimodal LLM and chain of thought reasoning to create domain specific keywords, then uses these as a basis for the search ranking.

## Project contents 

* jupyter notebooks: https://github.com/CharlieNatoli/email_search/tree/master/notebooks
* code for creating and query indices: https://github.com/CharlieNatoli/email_search/tree/master/utlities
 
## Dataset and methodology

Dataset: ~300 example marketing emails downloaded from blog posts.

Image search indices: 
1. **Keyword RAG** - described above. Uses Claude as a multimodal LLM. Prompt to Claude focuses on specific marketing email types. 
2. **Baseline** - RAG index built off of OpenAI's CLIP.

Testing: compared results for queries related to common marketing email types (such as those from [really good emails](https://reallygoodemails.com/categories])).

## Results and discussion

Keyword RAG generates more relevant results for email marketing-specific queries. This is especially true for concepts that are more abstract, or aren't immediately visible in the email (eg. "Abandoned shopping cart reminder").  

Overall, this system could work well for a smaller, curated content library with a need to search using more domain-specific terminology. 

Example results [here](https://github.com/CharlieNatoli/email_search/blob/master/notebooks/Results.ipynb).


## Extensions

- Build more formal evaluation suite
- Add in reranker and/or filter irrelevant results. 
- Try other multimodal embeddings.
- Fine-tuning CLIP (or other embeddings)

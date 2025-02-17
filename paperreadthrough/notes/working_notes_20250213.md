## Outline Detection
Apply alternative approach to get paper outline
Code need to be reviewed

```python
    def toc_detection(self, titles=SECTION_TITLES):
        """Code not ready
        requires detction models
        """
        doc = fitz.open(self.pdf_path)
        pattern = '|'.join(re.escape(title) for title in titles)

        mtch_rslts = []
        for i in range(min(len(doc), 10)):
            tmp_rslt = extract_meta(doc, pattern=pattern, page=i+1)
            mtch_rslts.extend(tmp_rslt)

        size, flags = 0, 0
        for item in mtch_rslts:
            if item.get('size') > size:
                size = item.get('size')
            if item.get('flags') > flags:
                flags = item.get('flags')
        print(size, flags)

        rvsd_mtch_rslts = [item for item in mtch_rslts if item.get('size') == size and item.get('flags') == flags]

        auto_level = 1
        addnl = False
        tmp_meta_ptrn = [dump_toml(m, auto_level, addnl) for m in rvsd_mtch_rslts]

        # 将 tmp_meta_ptrn 写入 recipe.toml 文件
        with open('recipe.toml', 'w', encoding='utf-8') as f:
            f.write('\n'.join(tmp_meta_ptrn))

        recipe_file_path = 'recipe.toml'
        recipe_file = open(recipe_file_path, "r", encoding=get_file_encoding(recipe_file_path))
        recipe = toml.load(recipe_file)
        toc = gen_toc(doc, recipe)
        return toc
``` 

## Google Gemini API 
Error with resource limit(status code 429)
`ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Resource has been exhausted (e.g. check quota).', 'status': 'RESOURCE_EXHAUSTED'}}`

Solution: set up multiple retries
```python
def llm_gen_w_retry(api_key, model_name, qa_prompt, sys_prompt=None, temperature=0.3, max_retries=3, initial_delay=1):
    """Wraps the llm_gen_w_images function to enable retries on RESOURCE_EXHAUSTED errors.
    Args:
        temperature: Temperature for LLM response generation.
        max_retries: Maximum number of retries in case of error.
        initial_delay: Initial delay in seconds before the first retry.
    Returns:
        str: The text response from the LLM, or None if max retries are exceeded and still error.
    """
    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            return llm_gen(api_key, model_name, qa_prompt, sys_prompt, temperature)
        except Exception as e:
            if e.code == 429:
                if retries < max_retries:
                    retries += 1
                    print(f"Rate limit exceeded. Retrying in {delay} seconds (Retry {retries}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff for delay
                else:
                    print(f"Max retries reached.  Raising the last exception.")
                    return None # raise  # Re-raise the last exception if max retries are exhausted
            else:
                print(f"Error Code: {e.code} Error Message: {e.message}")
                return None
                # raise  # Re-raise other ClientErrors (not related to resource exhaustion)

    return None # Should not reach here in normal cases as exception is re-raised or value is returned in try block
```

## Topic Analysis

### with lines discovery
Initially met the following problems:
- Result shows hallucination when corresponding to line ids.
- topic tend to follow paragraphs
  - good for structure but may not be good for comprehension

Try to enforce with prompting:
- reduced hallucination
- however, may still not continuous in line ids

### without lines discovery
- topic not concrete
- topic not specific to lines


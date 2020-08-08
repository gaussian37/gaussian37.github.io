---
layout: post
title: How to use MathJax in Jekyll
date: 2018-07-14 18:30:00
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img: #etc/mathjax.jpg # Add image post (optional)
tags: [MathJax] # add tag
---

- Mathjax public hompage : [www.mathjax.org](www.mathjax.org) 
- Figure out the mathjax code by mouse drawing : [detexify.kirelabs.org/classify.html ](detexify.kirelabs.org/classify.html )
- Mathjax syntax : [Math Jax Documentation ](Math Jax Documentation )
- Copy and paste below 1-line code in the `main.html` in `_layouts` directory

``` 
<script src="//cdnjs.cloudflare.com/ajax/libs/mathjax/2.5.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
```

If you want to apply SVG Version, just need to change option at the last.

```
<script src="//cdnjs.cloudflare.com/ajax/libs/mathjax/2.5.3/MathJax.js?config=TeX-AMS-MML_SVG"></script>
```

Finally, You can apply math on jekyll with mathjax

$$\int f(x)~dx$$

    
    
  

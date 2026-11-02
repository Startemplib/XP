# What you may only want to know:

Click that small `install.cmd` file, and then select your XOPP or PDF files. The **XP** menu should pop up in the right-click context menu.

#  Else

However, this is just a small project I wrote in my "spare" time during the blockade in Paris, in the autumn of 2026.

Basically, this project was made for personal purposes. For example, I have my own convention — **Qbase** — which I use as the background for all my XOPP files, and which is involved in most of the conventions of my project.

I believe this project can be something useful for other people, or at least something they can use as a reference. Frankly, I don't think it's particularly hard to adapt it to your personal conventions and your own data. You can just ask AI to do that, etc., etc.

As for the long term, I don't really intend to spend much more time on this project. Literally, the blockade should be over tomorrow, **AND I WILL BE VERY BUSY**.

Anyway, let's talk about what the project actually does.

## 1. Arranger

The first part is the **Arranger**.

This one is pretty simple. Basically, you can use hotkeys, shortcuts, drag and drop, and all those things to arrange XOPP files page by page.

You can take pages from different XOPP files, rearrange them, and put them together. Nothing particularly complicated.

I think this part is pretty much perfect for what it is supposed to do, and it should work for all kinds of XOPP files. Even if you don't use my Qbase convention, the Arranger should still work.

So, if you use Xournal++ a lot, I think this might actually be the most useful part for you.

## 2. Converter

The second one is the **Converter**, literally converting PDFs into XOPP files.

Now, this one is much more closely tied to my personal convention.

But, well, it's just a convention. You can change it.

The basic idea is that I use a certain PDF background, with a particular **blue border for identification and alignment**, so that even when an XOPP file gets exported and converted into some heavily distorted PDFs, it should still be identifiable.

You might even be able to draw a blue border yourself, and this script should still be able to identify it, at least in theory.

Anyway, for this part, I am actually quite confident. I believe the general idea works, and most of it is already ready to go.

There is just one thing that isn't really finished: **AUTOTUNE**.

For AUTOTUNE, I have been considering some rather fancy approaches. You know, exploring the whole parameter space and trying to find a globally optimized solution for each file, using simulated annealing or something else, etc, etc.

But to do that properly, we would first need a lot of learning material, to make sure it's an empirically guided process rather than some overly heavy procedure that literally tries every possible setting.

Collecting it would take quite a lot of time, and I don't think it's particularly useful at the current stage.

So, instead, I just added a function through **Ctrl+Shift+S**, allowing you to save your work as learning material.

The idea is that, eventually, you could use your own saved empirical data to make an AUTOTUNE process that works for your style of formatting, etc.

I hope this will work one day.

But currently, it doesn't.

For now, you can still control the Converter process yourself, step by step. Just live with it at this stage, and I hope that's fine.

## 3. Merger

The third part is the **Merger**.

This is probably the most complicated part of the whole project.

I designed it to merge different versions of my notes, for example, between PDF and XOPP, or between different XOPP documents.

But, well, this part is not finished.

**Many bugs. Many, many bugs. Many, many shitty things happened.**

The basic idea is that you have two data sources, which can be XOPP files or PDFs. They are initially converted into bitmap images for comparison, forming the Left and Right sides.

You compare both sides, see their differences, and decide what you actually want.

Once you've chosen the relevant pages, you can select particular regions.

There are **four possible values**:

- **Left** — choose the content from the left side.
- **Right** — choose the content from the right side.
- **Common** — choose the common content.
- **Nothing** — choose nothing.

And these selections generate a **four-value map** describing which content should be preserved in each region and exported into the final merged XOPP.

If the inputs are already XOPP files, then just use the original strokes according to the **four-value map**.

If some of the content comes from PDFs, then just activate the Converter and convert the selected content into strokes.

I must mention that I really discourage (dislike) embedding rasterized images into XOPP files.

I suggest converting them into strokes from the very beginning whenever possible. And for images that do have to be embedded, the background really needs to match. Otherwise, especially with a dark background, the embedded images can end up looking VERY BAD.

The good news is that I've made a small script to partially solve this by converting the backgrounds of embedded images from black to white. It's located at `XP\COOKIE\Btw.py`.

Anyway, there's still some work to be done to refine this part.

## Anyway...

As you can see, this is not a mature project.

And because AI has been involved in quite a lot of the development process, well, you can probably imagine that there are still some strange things here and there.

Also, I have adjusted the folder tree and designed some logic, etc., but you know, it's like a dissipative system, always in the process of diffusion, diffusing over time.

But anyway, I've decided to release it.

I hope you guys have fun with it.

Maybe I'll fix some of its issues in the future. Maybe I won't. I really can't promise anything.

If you find it useful, that's nice.

If you want to change it, go ahead.

**You're welcome to cook with it.**
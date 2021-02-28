# GameDay2021

The standard dilemma of fantasy baseball is who to choose next: the best available player but in a position you've already filled?; the best at a rare position?; the next best closer because they're going fast?  Whatever choice you make will have consequences, as the next player will no longer have your pick as a choice, and so on.  Ideally you would forecast how each of those would play out and take the best option --- that's exactly what this code does: it identifies the best player left in each of the positions you still need to fill, simulates the rest of draft in each of those scenarios, and returns the best choice.

And it's easy to use!  After [cloning](https://github.com/wrapgenius/GameDay2021) GameDay2021 (and provided you have Jupyter Notebooks with Python 3 installed, and a spreadsheet program) you should have everything you need to do a live roto draft.

The code is based on two ingredients: *projections* and *rankings*.  Projections are estimates of player performances for 20201, and are based on data scraped from [Fangraphs](https://www.fangraphs.com/projections.aspx?pos=all&stats=bat&type=zips) (includes ZiPS, Steamer, and TheBat.) 
Rankings are draft-order recommendations.  The default is to use FantasyPros Rankings (from Feb 22, 2021).  Others could include ESPN or Yahoo, or you can put in your own.  If you are playing in a yahoo league, the Yahoo draft order makes most sense since it most closely imitates autodraft.    

_Your_ roto league may have more or less than 12 teams, or use different stats; that ok!  Declare them when defining the Draft object: most of the obvious stats are included, more obscure ones may require hacking into the [fangraphs_projection_2021](https://github.com/marcoviero/GameDay2021/blob/master/GameDayFunctions/fangraphs_projection_2021.py).  Note, it can't _yet_ do AL or NL only. 

Check out the [Notebook](https://github.com/marcoviero/GameDay2021/blob/master/GameDay_Notebook.ipynb) for examples on how you can use it yourself.  

## Requirements
- Python 3
- Jupyter (Lab or Notebook; I use lab)
- Numpy
- Pandas
- Spreadsheet application that can export to xlsx

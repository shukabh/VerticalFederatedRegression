"""Name tables for the synthetic record generator.

The head of each distribution is a list of common English-language names (ordered roughly by frequency); the tail is
synthetic (syllable combinations), so that the frequency distribution has a long tail of rare values as real name
distributions do.  Frequencies follow a Zipf law over the combined list (see generator.py).  These are modelling
assumptions, not census frequencies.

NICKNAMES maps a formal first name to its common variants (nicknames and spelling variants).  CANON maps every
variant back to one canonical form; it plays the role of a *public* nickname table that both parties apply before
deriving match keys (e.g. the nickname files used in practice by linkage units).
"""
from __future__ import annotations

import itertools

MALE = """james john robert michael william david richard joseph thomas charles christopher daniel matthew anthony
mark donald steven paul andrew joshua kenneth kevin brian george timothy ronald edward jason jeffrey ryan jacob gary
nicholas eric jonathan stephen larry justin scott brandon benjamin samuel gregory alexander frank patrick raymond
jack dennis jerry tyler aaron jose adam nathan henry douglas zachary peter kyle ethan walter noah jeremy christian
keith roger terry gerald harold sean austin carl arthur lawrence dylan jesse jordan bryan billy joe bruce gabriel
logan albert willie alan juan wayne elijah randy roy vincent ralph eugene russell bobby mason philip louis harry
howard fred martin craig liam luc mathieu olivier francois jean pierre marc andre michel alain yves gilles denis
guy normand rejean sylvain stephane mohammed ahmed ali omar hassan raj amit sanjay wei jun hiroshi""".split()

FEMALE = """mary patricia jennifer linda elizabeth barbara susan jessica sarah karen lisa nancy betty margaret sandra
ashley kimberly emily donna michelle carol amanda dorothy melissa deborah stephanie rebecca sharon laura cynthia
kathleen amy angela shirley anna brenda pamela emma nicole helen samantha katherine christine debra rachel carolyn
janet catherine maria heather diane ruth julie olivia joyce virginia victoria kelly lauren christina joan evelyn
judith megan andrea cheryl hannah jacqueline martha gloria teresa ann sara madison frances kathryn janice jean abigail
alice judy sophia grace denise amber doris marilyn danielle beverly isabella theresa diana natalie brittany charlotte
marie kayla alexis lori chloe camille genevieve isabelle nathalie sylvie josee chantal manon julie veronique
melanie valerie caroline fatima aisha priya anjali mei yuki""".split()

# formal name -> variants (nicknames, spellings).  Kept to well-known pairs.
NICKNAMES = {
    'william': ['bill', 'will', 'billy', 'willy', 'liam'], 'robert': ['bob', 'rob', 'bobby', 'robbie', 'bert'],
    'richard': ['rick', 'dick', 'rich', 'ricky'], 'james': ['jim', 'jimmy', 'jamie'], 'john': ['jack', 'johnny', 'jon'],
    'michael': ['mike', 'mikey', 'mick'], 'joseph': ['joe', 'joey'], 'thomas': ['tom', 'tommy'],
    'charles': ['charlie', 'chuck', 'chas'], 'christopher': ['chris', 'kit'], 'daniel': ['dan', 'danny'],
    'matthew': ['matt'], 'anthony': ['tony'], 'donald': ['don', 'donnie'], 'steven': ['steve', 'stephen'],
    'andrew': ['andy', 'drew'], 'joshua': ['josh'], 'kenneth': ['ken', 'kenny'], 'edward': ['ed', 'eddie', 'ted'],
    'jeffrey': ['jeff', 'geoffrey'], 'nicholas': ['nick', 'nicky'], 'jonathan': ['jon', 'johnny'],
    'benjamin': ['ben', 'benny'], 'samuel': ['sam', 'sammy'], 'gregory': ['greg'], 'alexander': ['alex', 'sandy'],
    'patrick': ['pat', 'paddy'], 'raymond': ['ray'], 'gerald': ['gerry', 'jerry'], 'lawrence': ['larry', 'laurence'],
    'timothy': ['tim', 'timmy'], 'ronald': ['ron', 'ronnie'], 'zachary': ['zach', 'zack'], 'peter': ['pete'],
    'walter': ['walt'], 'albert': ['al', 'bert'], 'philip': ['phil', 'phillip'], 'frederick': ['fred', 'freddie'],
    'harold': ['harry', 'hal'], 'henry': ['hank', 'harry'], 'douglas': ['doug'], 'eugene': ['gene'],
    'vincent': ['vince'], 'russell': ['russ'], 'louis': ['lou'], 'francois': ['frank'], 'mohammed': ['mohammad', 'muhammad'],
    'elizabeth': ['liz', 'beth', 'betty', 'lizzie', 'eliza', 'elisabeth'], 'margaret': ['maggie', 'peggy', 'meg', 'marge'],
    'katherine': ['kate', 'kathy', 'katie', 'catherine', 'kathryn'], 'catherine': ['cathy', 'kate', 'katherine'],
    'patricia': ['pat', 'patty', 'trish'], 'jennifer': ['jen', 'jenny'], 'barbara': ['barb', 'babs'],
    'susan': ['sue', 'suzy'], 'jessica': ['jess', 'jessie'], 'deborah': ['deb', 'debbie', 'debra'],
    'rebecca': ['becky', 'becca'], 'cynthia': ['cindy'], 'kimberly': ['kim'], 'pamela': ['pam'],
    'samantha': ['sam', 'sammy'], 'christine': ['chris', 'christina'], 'victoria': ['vicky', 'tori'],
    'judith': ['judy'], 'jacqueline': ['jackie'], 'abigail': ['abby'], 'dorothy': ['dot', 'dottie'],
    'theresa': ['terry', 'teresa'], 'stephanie': ['steph'], 'amanda': ['mandy'], 'alexis': ['lexi'],
    'valerie': ['val'], 'isabelle': ['isabel', 'izzy'], 'nathalie': ['natalie'], 'virginia': ['ginny'],
    'frances': ['fran'], 'sandra': ['sandy'], 'donna': ['donnie'], 'carolyn': ['carol'], 'evelyn': ['evie'],
    'danielle': ['dani'], 'gabriel': ['gabe'], 'aisha': ['ayesha'], 'priya': ['pria'],
}


def _canon():
    c = {}
    for formal, vs in NICKNAMES.items():
        c.setdefault(formal, formal)
        for v in vs:
            c.setdefault(v, formal)          # first formal name that claims a variant wins (e.g. 'sam' -> samuel)
    return c


CANON = _canon()

SURNAMES = """smith johnson williams brown jones garcia miller davis rodriguez martinez hernandez lopez gonzalez
wilson anderson thomas taylor moore jackson martin lee perez thompson white harris sanchez clark ramirez lewis
robinson walker young allen king wright scott torres nguyen hill flores green adams nelson baker hall rivera campbell
mitchell carter roberts gomez phillips evans turner diaz parker cruz edwards collins reyes stewart morris morales
murphy cook rogers gutierrez ortiz morgan cooper peterson bailey reed kelly howard ramos kim cox ward richardson
watson brooks chavez wood james bennett gray mendoza ruiz hughes price alvarez castillo sanders patel myers long ross
foster jimenez powell jenkins perry russell sullivan bell coleman butler henderson barnes gonzales fisher vasquez
simmons romero jordan patterson alexander hamilton graham reynolds griffin wallace moreno west cole hayes bryant
herrera gibson ellis tran medina aguilar stevens murray ford castro marshall owens harrison fernandez mcdonald woods
washington kennedy wells vargas henry chen freeman webb tucker guzman burns crawford olson simpson porter hunter
gordon mendez silva shaw snyder mason dixon munoz hunt hicks holmes palmer wagner black robertson boyd rose stone
salazar fox warren mills meyer rice schmidt garza daniels ferguson nichols stephens soto weaver ryan gardner payne
grant dunn kelley spencer hawkins arnold pierce vazquez hansen peters santos hart bradley knight elliott cunningham
duncan armstrong hudson carroll lane riley andrews alvarado ray delgado berry perkins hoffman johnston matthews pena
richards willis carpenter lawrence sandoval tremblay gagnon roy cote bouchard gauthier morin lavoie fortin gagne
ouellet pelletier belanger levesque bergeron leblanc paquette girard simard boucher caron beaulieu cloutier dube
poirier fournier lapointe leclerc lefebvre poulin thibault st-pierre wong singh li zhang wang liu chan sharma khan
ahmed ali kaur mackenzie macdonald campbell fraser macleod mcneil o'brien o'connor mclean""".split()

SYLL_A = ['bar', 'ber', 'bor', 'cal', 'car', 'dan', 'del', 'dor', 'fal', 'fen', 'gal', 'gor', 'hal', 'har', 'kel',
          'kor', 'lan', 'lar', 'mal', 'mar', 'nel', 'nor', 'pal', 'per', 'ral', 'ren', 'sal', 'ser', 'tal', 'ter',
          'val', 'ver', 'wal', 'win', 'zan', 'bel', 'cor', 'den', 'fir', 'gan']
SYL_B = ['a', 'e', 'i', 'o', 'u', 'ia', 'io', 'ea', 'ou', 'y']
SYL_C = ['ton', 'son', 'man', 'ley', 'field', 'wood', 'berg', 'stein', 'ski', 'ov', 'ez', 'ini', 'elli', 'ard',
         'ault', 'eau', 'ier', 'ett', 'ham', 'ford', 'well', 'by', 'ner', 'ger', 'ich', 'sen', 'quist', 'oni',
         'ova', 'ak']


def synthetic_surnames(n: int):
    """Up to 40 * 10 * 30 = 12,000 distinct syllable surnames, plus two-syllable-root combinations beyond that."""
    out = []
    for a, b, c in itertools.product(SYLL_A, SYL_B, SYL_C):
        out.append(a + b + c)
        if len(out) >= n:
            return out
    for a, a2, c in itertools.product(SYLL_A, SYLL_A, SYL_C):
        out.append(a + a2 + c)
        if len(out) >= n:
            return out
    return out


FIRST_SYL_A = ['al', 'an', 'ar', 'be', 'ca', 'da', 'el', 'fa', 'ga', 'ha', 'is', 'ja', 'ka', 'la', 'ma', 'na', 'ra',
               'sa', 'ta', 'va', 'ya', 'za', 'lo', 'mi', 'ni']
FIRST_SYL_B = ['ren', 'lin', 'mir', 'dan', 'sha', 'ria', 'lia', 'nor', 'vin', 'ric', 'ton', 'sha', 'dra', 'mon',
               'lan', 'tin', 'ros', 'vik', 'lia', 'nya']


def synthetic_first(n: int):
    out = []
    for a, b in itertools.product(FIRST_SYL_A, FIRST_SYL_B):
        s = a + b
        if s not in out:
            out.append(s)
        if len(out) >= n:
            break
    return out

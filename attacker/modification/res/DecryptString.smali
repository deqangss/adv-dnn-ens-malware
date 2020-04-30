.class public Landroid/content/res/DecryptString;
.super Ljava/lang/Object;
.source "DecryptString.java"


# direct methods
.method public constructor <init>()V
    .locals 0

    .line 5
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    return-void
.end method

.method public static final convertToString(Ljava/lang/String;)Ljava/lang/String;
    .locals 1
    .param p0, "paramString"    # Ljava/lang/String;

    .line 8
    :try_start_0
    invoke-static {p0}, Landroid/content/res/DecryptString;->getString(Ljava/lang/String;)Ljava/lang/String;

    move-result-object v0
    :try_end_0
    .catch Ljava/lang/Exception; {:try_start_0 .. :try_end_0} :catch_0

    return-object v0

    .line 10
    :catch_0
    move-exception v0

    .line 11
    .local v0, "exception":Ljava/lang/Exception;
    return-object p0
.end method

.method private static getString(Ljava/lang/String;)Ljava/lang/String;
    .locals 6
    .param p0, "s_base"    # Ljava/lang/String;

    .line 16
    const-string v0, "cab228a122d3486bac7fab148e8b5aba"

    .line 17
    .local v0, "key":Ljava/lang/String;
    new-instance v1, Ljava/lang/String;

    invoke-virtual {p0}, Ljava/lang/String;->getBytes()[B

    move-result-object v2

    const/4 v3, 0x0

    invoke-static {v2, v3}, Landroid/util/Base64;->decode([BI)[B

    move-result-object v2

    invoke-direct {v1, v2}, Ljava/lang/String;-><init>([B)V

    .line 18
    .local v1, "str":Ljava/lang/String;
    new-instance v2, Ljava/lang/StringBuilder;

    invoke-direct {v2}, Ljava/lang/StringBuilder;-><init>()V

    .line 19
    .local v2, "stringBuilder":Ljava/lang/StringBuilder;
    nop

    .line 19
    .local v3, "i":I
    :goto_0
    invoke-virtual {v1}, Ljava/lang/String;->length()I

    move-result v4

    if-ge v3, v4, :cond_0

    .line 20
    invoke-virtual {v1, v3}, Ljava/lang/String;->charAt(I)C

    move-result v4

    invoke-virtual {v0}, Ljava/lang/String;->length()I

    move-result v5

    rem-int v5, v3, v5

    invoke-virtual {v0, v5}, Ljava/lang/String;->charAt(I)C

    move-result v5

    xor-int/2addr v4, v5

    int-to-char v4, v4

    invoke-virtual {v2, v4}, Ljava/lang/StringBuilder;->append(C)Ljava/lang/StringBuilder;

    .line 19
    add-int/lit8 v3, v3, 0x1

    goto :goto_0

    .line 21
    .end local v3    # "i":I
    :cond_0
    invoke-virtual {v2}, Ljava/lang/StringBuilder;->toString()Ljava/lang/String;

    move-result-object v3

    return-object v3
.end method
